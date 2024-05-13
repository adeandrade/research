from typing import Sequence, Tuple

from torch import Tensor


TargetTypeDetection = Tuple[Tensor, Tensor, Tensor]
TargetTypeDetectionSegmentation = Tuple[TargetTypeDetection, Tensor]
BatchTypeDetectionSegmentation = Tuple[Sequence[Tensor], Sequence[TargetTypeDetectionSegmentation]]
BatchTypeDetection = Tuple[Sequence[Tensor], Sequence[TargetTypeDetection]]
BatchTypeSegmentationDepth = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]
BatchTypeSegmentation = Tuple[Tensor, Tensor]
BatchTypeDepth = Tuple[Tensor, Tuple[Tensor, Tensor]]
BatchTypeMaskedTensor = Tuple[Tensor, Tensor]
BatchTypeTensors = Sequence[Tensor]
