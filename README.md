# AI-Hand-Gesture-Painting
The app captures live video from your webcam and detects your hand using MediaPipe’s hand tracking technology. If index finger is fully extended it activates "painting mode" and draws lines as you move your finger through the air. The rest of the hand must remain closed, only one fully open finger is allowed to draw.
How it works:
When you run the program, your webcam opens and starts tracking your hand using MediaPipe. The app continuously checks your hand posture: if only your index finger is fully extended (and all other fingers are folded), it activates "painting mode." As you move your finger, it draws lines on a separate canvas window in real time. If you make a fist or raise more than one finger, the painting stops automatically. You can change the drawing color anytime using keyboard keys: r (red), g (green), b (blue), or y (yellow). Press Esc to exit. Simple, hands-free drawing — just wave your finger and paint!

![Image](https://github.com/user-attachments/assets/51a432a5-62f0-4dce-a15b-c0b688009a00)
