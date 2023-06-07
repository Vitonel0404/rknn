import torch
from torchvision.ops.boxes import box_iou

def nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]

if __name__ == '__main__':
    # ... aquí va el código de cargar la imagen/video. Asumamos que ya estamos en inferencia:

    outputs = rknn.inference(inputs=[img])

    # Extract boxes (elements 0, 1, 2, 3) from the second dimension
    bboxes = outputs[0, :, 0:4].astype(int)  # Shape: (25200, 4)

    # Extract scores (element 4) from the second dimension
    scores = outputs[0, :, 4:5]  # Shape: (25200, 1)

    # Extract cls (element 5) from the second dimension
    classes = outputs[0, :, 5:6].astype(int)  # Shape: (25200, 1)

    indices = nms(bboxes=torch.tensor(bboxes), scores=torch.tensor(scores), iou_threshold=0.5)

    # Estos son los resultados que buscas:
    bboxes = bboxes[indices]
    scores = scores[indices]
    classes = classes[indices]

    
