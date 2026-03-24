from collections.abc import Callable

import defopt
import mlflow
import numpy as np
import scipy.sparse.linalg as linalg_scipy
import sfu_torch_lib.mlflow as mlflow_lib
import torch
import torch.autograd.functional as functional_ag
import torch.linalg as linalg_torch
import torch.nn.functional as functional_nn
import tqdm
from scipy.sparse.linalg import LinearOperator
from sfu_torch_lib import slack, state
from torch import Generator, Tensor
from torch.autograd import Variable
from torch.nn import attention
from torch.nn.attention import SDPBackend
from torch.utils.data import DataLoader, RandomSampler

import lm_codec.analyze_rate_distortion as rd
from lm_codec import functions
from lm_codec.dataset_openwebtext import OpenWebText, OpenWebTextReTokenize
from lm_codec.model_lm_codec import LMCodec
from lm_codec.model_lm_codec_litgpt import LMCodecAdHoc as LMCodecAdHocLitGPT

LMType = LMCodecAdHocLitGPT | LMCodec


class Operator:
    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[Tensor], Tensor],
        rmatvec: Callable[[Tensor], Tensor],
        get_row: Callable[[int], Tensor],
        get_column: Callable[[int], Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.shape = shape
        self.matvec = matvec
        self.rmatvec = rmatvec
        self.get_row = get_row
        self.get_column = get_column
        self.dtype = dtype
        self.device = device

    def transpose(self) -> 'Operator':
        return Operator(
            self.shape,
            self.rmatvec,
            self.matvec,
            self.get_column,
            self.get_row,
            self.dtype,
            self.device,
        )

    def dot_transpose(self) -> 'Operator':
        return Operator(
            shape=self.shape,
            matvec=lambda x: self.rmatvec(self.matvec(x)),
            rmatvec=lambda x: self.rmatvec(self.matvec(x)),
            get_row=lambda i: self.rmatvec(self.get_column(i)),
            get_column=lambda i: self.rmatvec(self.get_column(i)),
            dtype=self.dtype,
            device=self.device,
        )

    def to_scipy(self) -> LinearOperator:
        return LinearOperator(
            shape=self.shape,
            matvec=lambda x: functions.to_numpy(  # type: ignore
                self.matvec(
                    torch.as_tensor(
                        x,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            ),
            rmatvec=lambda x: functions.to_numpy(  # type: ignore
                self.rmatvec(
                    torch.as_tensor(
                        x,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            ),
            dtype=np.float32,
        )

    def inverse(self) -> 'Operator':
        return Operator(
            shape=self.shape,
            matvec=lambda x: solve_ls(self.to_scipy(), x),
            rmatvec=lambda x: solve_ls(self.transpose().to_scipy(), x),
            get_row=lambda i: solve_ls(self.to_scipy(), self.get_row(i)),
            get_column=lambda i: solve_ls(self.to_scipy(), self.get_column(i)),
            dtype=self.dtype,
            device=self.device,
        )


def create_index_vector(index: int, inputs: Tensor) -> Tensor:
    vector = torch.zeros([torch.numel(inputs)], dtype=inputs.dtype, device=inputs.device)
    vector[index] = 1
    return torch.reshape(vector, inputs.shape)


def create_simple_operator(matrix: Tensor) -> Operator:
    return Operator(
        shape=(matrix.shape[0], matrix.shape[1]),
        matvec=lambda x: matrix @ x,
        rmatvec=lambda x: matrix.T @ x,
        get_row=lambda index: matrix[index],
        get_column=lambda index: matrix[:, index],
        dtype=matrix.dtype,
        device=matrix.device,
    )


def create_jacobian_operator(
    function: Callable[[Tensor], Tensor],
    inputs: Tensor,
    create_graph: bool = False,
) -> Operator:
    num_elements = torch.numel(inputs)

    return Operator(
        shape=(num_elements, num_elements),
        matvec=lambda x: torch.flatten(
            functional_ag.jvp(
                function,
                inputs,
                torch.reshape(x, inputs.shape),
                create_graph=create_graph,
            )[1]  # type: ignore
        ),
        rmatvec=lambda x: torch.flatten(
            functional_ag.vjp(
                function,
                inputs,
                torch.reshape(x, inputs.shape),
                create_graph=create_graph,
            )[1]  # type: ignore
        ),
        get_row=lambda index: torch.flatten(
            functional_ag.vjp(
                function,
                inputs,
                create_index_vector(index, inputs),
            )[1],  # type: ignore
        ),
        get_column=lambda index: torch.flatten(
            functional_ag.jvp(
                function,
                inputs,
                create_index_vector(index, inputs),
            )[1],  # type: ignore
        ),
        dtype=inputs.dtype,
        device=inputs.device,
    )


def jacobian_operator_creator(
    function: Callable[[Tensor, Tensor | None], Tensor],
    inputs: Tensor,
    create_graph: bool = False,
) -> Callable[[Tensor | None], Operator]:
    def _operator_creator(mu: Tensor | None = None):
        return create_jacobian_operator(lambda inputs: function(inputs, mu), inputs, create_graph)

    return _operator_creator


def create_random_vector(operator: Operator) -> Tensor:
    return torch.randn(operator.shape[1:2], dtype=operator.dtype, device=operator.device)


def create_unit_vector(operator: Operator) -> Tensor:
    vector = create_random_vector(operator)
    return vector / torch.norm(vector)


def create_ones_vector(operator: Operator) -> Tensor:
    return torch.ones(operator.shape[1:2], dtype=operator.dtype, device=operator.device)


def calculate_spectral_norm(operator: Operator, num_iterations: int, v: Tensor | None = None) -> tuple[float, Tensor]:
    v = create_unit_vector(operator) if v is None else v

    operator = operator.dot_transpose()

    for _ in range(num_iterations):
        v = operator.matvec(v)
        v = functional_nn.normalize(v, dim=0)

    spectral_radius = operator.matvec(v)
    spectral_radius = torch.dot(spectral_radius, v)
    spectral_radius = torch.sqrt(spectral_radius)
    spectral_radius = torch.log(spectral_radius)
    spectral_radius = spectral_radius.item()

    return spectral_radius, v


def calculate_error_torch(operator: Operator, x: Tensor, b: Tensor) -> float:
    return torch.max(torch.abs(operator.matvec(x) - b)).item()


def calculate_error_scipy(operator: LinearOperator, x: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(operator.matvec(x) - b)).item()


def solve_gd(
    operator: Operator,
    b: Tensor,
    x: Tensor | None = None,
    learning_rate: float = 1e-3,
    tolerance: float = 1e-2,
    max_iterations: int = 100,
) -> Tensor:
    x = b if x is None else x

    variables = Variable(x, requires_grad=True)

    for _ in range(max_iterations):
        loss = operator.matvec(variables)
        loss = 1 - functional_nn.cosine_similarity(loss, b, 0)

        # print(calculate_error_torch(operator, variables, b))
        if loss <= tolerance:
            break

        loss.backward(retain_graph=True)

        assert variables.grad is not None

        variables.data = variables.data - learning_rate * variables.grad.data
        variables.grad = None

    return torch.detach(variables.data)


def solve_cgs(
    operator: LinearOperator,
    b: Tensor,
    x: Tensor | None = None,
    tolerance: float = 1e-5,
    max_iterations: int | None = 100,
) -> Tensor:
    x0 = b if x is None else x

    solution, *_ = linalg_scipy.cgs(
        operator,
        functions.to_numpy(b),
        functions.to_numpy(x0),
        rtol=tolerance,
        maxiter=max_iterations,
    )

    # print(calculate_error_scipy(operator, solution, functions.to_numpy(b)))
    return torch.tensor(solution, dtype=b.dtype, device=b.device)


def solve_gmres(
    operator: LinearOperator,
    b: Tensor,
    x: Tensor | None = None,
    tolerance: float = 1e-5,
    max_iterations: int | None = 100,
) -> Tensor:
    x0 = b if x is None else x

    solution, *_ = linalg_scipy.gmres(
        operator,
        functions.to_numpy(b),
        functions.to_numpy(x0),
        rtol=tolerance,
        maxiter=max_iterations,
    )

    # print(calculate_error_scipy(operator, x, functions.to_numpy(b)))
    return torch.tensor(solution, dtype=b.dtype, device=b.device)


def solve_ls(
    operator: LinearOperator,
    b: Tensor,
    x: Tensor | None = None,
    tolerance: float = 1e-5,
    max_iterations: int | None = 100,
) -> Tensor:
    x0 = b if x is None else x

    solution, *_ = linalg_scipy.lsmr(
        operator,
        functions.to_numpy(b),
        x0=functions.to_numpy(x0),
        atol=tolerance,
        btol=tolerance,
        maxiter=max_iterations,
    )

    # print(calculate_error_scipy(operator, solution, functions.to_numpy(b)))
    return torch.tensor(solution, dtype=b.dtype, device=b.device)


def solve_kaczmarz(
    operator: Operator,
    b: Tensor,
    x: Tensor | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 1000,
) -> Tensor:
    x = b if x is None else x

    assert x is not None

    h, _ = operator.shape

    for k in range(max_iterations):
        i = k % h
        a_i = operator.get_row(i)
        b_i = b[i]

        x = x + (b_i - torch.dot(a_i, x)) / (torch.norm(a_i) ** 2) * a_i  # type: ignore

        error = calculate_error_torch(operator, x, b)  # type: ignore

        # print(error)
        if error < tolerance:
            break

    return x  # type: ignore


def calculate_spectral_radius_inverse(
    create_operator: Callable[..., Operator],
    num_iterations: int,
    v: Tensor | None = None,
) -> tuple[float, Tensor]:
    operator = create_operator()

    v = create_unit_vector(operator) if v is None else v
    # mu = torch.zeros_like(inputs)
    mu = None

    operator = create_operator(mu)

    operator_scipy = operator.dot_transpose().to_scipy()

    for _ in range(num_iterations):
        v = solve_ls(operator_scipy, v)
        v = functional_nn.normalize(v, dim=0)

        # mu = operator.rmatvec(v)
        # mu = torch.reshape(mu, inputs.shape)
        # operator = create_operator_scipy(lambda x: function(x, mu), inputs)

    spectral_radius = solve_ls(operator_scipy, v)
    spectral_radius = torch.dot(spectral_radius, v)
    spectral_radius = spectral_radius.item()

    return spectral_radius, v


def arnoldi_iteration(operator: Operator, x: Tensor, n: int, epsilon: float = 1e-12) -> tuple[Tensor, Tensor]:
    """
    Compute a basis of the (n + 1)-Krylov subspace of the matrix A.
    This is the space spanned by the vectors {b, Ab, ..., A^n b}.
    """
    _, width = operator.shape

    krylov = torch.zeros((width, n + 1), dtype=x.dtype, device=x.device)
    hessenberg = torch.zeros((n + 1, n), dtype=x.dtype, device=x.device)

    # normalize the input vector, use it as the first Krylov vector
    krylov[:, 0] = x / torch.norm(x)

    for k in range(1, n + 1):
        # generate a new candidate vector
        vector = operator.matvec(krylov[:, k - 1])

        # subtract the projections on previous vectors
        for j in range(k):
            hessenberg[j, k - 1] = torch.dot(krylov[:, j], vector)
            vector = vector - hessenberg[j, k - 1] * krylov[:, j]

        hessenberg[k, k - 1] = torch.norm(vector)

        # add the produced vector to the list, unless
        if hessenberg[k, k - 1] > epsilon:
            krylov[:, k] = vector / hessenberg[k, k - 1]

        # if that happens, stop iterating
        else:
            break

    krylov = krylov[:, :k]
    hessenberg = hessenberg[:k, :k]

    return krylov, hessenberg


def calculate_determinant(eigenvalues: Tensor) -> float:
    determinant = torch.log(eigenvalues)
    determinant = torch.mean(determinant)
    return determinant.item()


def calculate_determinant_torch(operator: Operator, n: int) -> float:
    vector = create_unit_vector(operator)

    _, hessenberg = arnoldi_iteration(operator, vector, n)

    eigenvalues = linalg_torch.eigvals(hessenberg)
    eigenvalues = torch.abs(eigenvalues)

    return calculate_determinant(eigenvalues)


def calculate_determinant_torch_singular(operator: Operator, n: int) -> float:
    operator = operator.dot_transpose()

    vector = create_unit_vector(operator)

    _, hessenberg = arnoldi_iteration(operator, vector, n)

    eigenvalues = linalg_torch.eigvals(hessenberg)
    eigenvalues = torch.abs(eigenvalues)
    eigenvalues = torch.sqrt(eigenvalues)

    return calculate_determinant(eigenvalues)


def calculate_determinant_scipy(operator: Operator, n: int) -> float:
    operator_scipy = operator.to_scipy()

    eigenvalues = linalg_scipy.eigs(
        operator_scipy,
        n,
        return_eigenvectors=False,
    )  # type: ignore

    eigenvalues = torch.tensor(eigenvalues, dtype=operator.dtype, device=operator.device)
    eigenvalues = torch.abs(eigenvalues)
    eigenvalues = torch.sqrt(eigenvalues)

    return calculate_determinant(eigenvalues)


def calculate_determinant_symmetric(operator: Operator, n: int) -> float:
    operator = operator.dot_transpose()

    vector = create_unit_vector(operator)

    _, hessenberg = arnoldi_iteration(operator, vector, n)
    eigenvalues_largest = linalg_torch.eigvals(hessenberg)
    eigenvalues_largest = torch.abs(eigenvalues_largest)
    eigenvalues_largest = torch.sqrt(eigenvalues_largest)

    operator_inverse = operator.inverse()
    _, hessenberg = arnoldi_iteration(operator_inverse, vector, n)
    eigenvalues_smallest = linalg_torch.eigvals(hessenberg)
    eigenvalues_smallest = torch.abs(eigenvalues_smallest)
    eigenvalues_smallest = torch.sqrt(eigenvalues_smallest)
    eigenvalues_smallest = 1 / eigenvalues_smallest

    determinant = torch.stack((eigenvalues_largest, eigenvalues_smallest))
    return calculate_determinant(determinant)


def calculate_log_determinant_power_hall(create_operator: Callable[..., Operator], num_iterations: int) -> float:
    operator = create_operator()

    mu = create_ones_vector(operator)
    v = u = create_random_vector(operator)

    operator = create_operator(mu)

    log_determinant, sign = 0, -1

    for index in range(num_iterations):
        u = operator.rmatvec(u)
        sign *= -1
        log_determinant += (sign / (index + 1) * torch.dot(u, v)).item()

    return log_determinant


def calculate_trace(operator: Operator, num_iterations: int) -> float:
    trace = 0

    for index in range(num_iterations):
        value = create_random_vector(operator)
        value = operator.matvec(value)
        value = torch.dot(value, value)

        trace += (value.item() - trace) / (index + 1)

    return trace


def calculate_trace_inverse(operator: Operator, num_iterations: int) -> float:
    operator_scipy = operator.to_scipy()

    trace = 0

    for index in range(num_iterations):
        value = create_random_vector(operator)
        value = solve_ls(operator_scipy, value)
        value = torch.dot(value, value)

        trace += (value.item() - trace) / (index + 1)

    return trace


def calculate_expectation(function: Callable[[Tensor], float], dataloader: DataLoader) -> float:
    expectation = 0

    for index, inputs in tqdm.tqdm(enumerate(iter(dataloader))):
        with attention.sdpa_kernel(SDPBackend.MATH):
            value = function(inputs)

        expectation += (value - expectation) / (index + 1)
        # print(expectation)

    return expectation


def calculate_expectation_v(
    function: Callable[[Tensor, Tensor | None], tuple[float, Tensor]],
    dataloader: DataLoader,
) -> float:
    expectation = 0

    v = None

    for index, inputs in tqdm.tqdm(enumerate(iter(dataloader))):
        with attention.sdpa_kernel(SDPBackend.MATH):
            value, v = function(inputs, v)

        expectation += (value - expectation) / (index + 1)
        # print(expectation)

    return expectation


@torch.no_grad
def transform(x: Tensor, model: LMType) -> Tensor:
    *_, (y, *_) = model.lm.forward(x, return_blocks={model.split_index}, quantize=True)

    return y


@torch.compiler.set_stance('force_eager')
def predict(x: Tensor, model: LMType, mu: Tensor | None = None) -> Tensor:
    parameters = model.analysis_prior.forward(x)
    parameters = functions.quantize(parameters)
    parameters = model.synthesis_prior.forward(parameters)

    likelihoods = -1 * model.model_representation.nll_discrete(x, parameters)
    return likelihoods if mu is None else likelihoods - mu * x


@slack.notify
@mlflow_lib.install
def analyze(
    *,
    run_id_pretrained: str,
    run_id: str | None = None,
    dataset_type: str = 'openwebtext',
    num_documents: int = 100,
    block_size: int = 512,
    num_iterations: int = 1000,
    num_steps: int = 10,
    seed: int = 110069,
) -> None:
    """
    Compute spectral norm of the entropy model.

    :param run_id_pretrained: run ID of the training run
    :param run_id: run ID of the current run
    :param dataset_type: dataset name
    :param num_documents: number of samples
    :param block_size: context size
    :param num_iterations: number of power iterations
    :param num_steps: number of repeated runs for error analysis
    :param seed: random seed
    """
    model = state.load_model(run_id_pretrained, cache=True, overwrite=False)
    assert isinstance(model, LMType)
    model = model.eval()

    def dataset_transform(batch: tuple[Tensor, Tensor]) -> Tensor:
        inputs, _ = batch
        inputs = inputs.cuda()
        return transform(inputs[None], model)[0]

    if dataset_type == 'openwebtext':
        if hasattr(model, 'tokenizer'):
            assert isinstance(model, LMCodecAdHocLitGPT)

            dataset = OpenWebTextReTokenize(model.tokenizer, 'validation', block_size, transform=dataset_transform)

        else:
            dataset = OpenWebText('validation', block_size, transform=dataset_transform)

    else:
        raise ValueError(f'Validation dataset {dataset_type} not supported.')

    generator = Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=RandomSampler(dataset, False, num_documents, generator),
    )

    loss, bpt, distortion = rd.find_best_metrics(
        run_id=run_id_pretrained,
        calculate_loss=lambda loss, *_: loss,
        metric_labels=['Validation Loss', 'Validation BPT', 'Validation Distortion'],
    )

    mlflow.log_metrics({
        'Index': model.split_index,
        'Loss': loss,
        'BPT': bpt,
        'Distortion': distortion,
    })

    for step in range(num_steps):
        value = calculate_expectation_v(
            lambda inputs, v: calculate_spectral_norm(
                create_jacobian_operator(
                    lambda x: predict(x, model),
                    inputs,
                ),
                num_iterations,
                v,
            ),
            dataloader,
        )

        mlflow.log_metrics(
            {
                'Index': model.split_index,
                'Spectral Norm': value,
                'Loss': loss,
                'BPT': bpt,
                'Distortion': distortion,
            },
            step=step,
        )


def main():
    defopt.run(analyze)


if __name__ == '__main__':
    main()
