from ultralytics import YOLO
import cv2
import torch

labels = {0: 'prone', 1: 'sideways', 2: 'suspine'}

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # this is a list of Results (usually length 1)
    result = results[0]     # get the first Results object

    # For classification models, 'probs' contains class probabilities (Tensor)
    # Check if 'probs' exists
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs  # Tensor of shape [1, num_classes]
        pred_class = torch.argmax(probs, dim=1).item()
        label = labels.get(pred_class, 'Unknown')
    else:
        label = 'No prediction'

    cv2.putText(frame, f'Pose: {label}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Baby Pose Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
