from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingHistory:
    losses: list[float]
    accuracies: list[float]


class SimpleRNNClassifier:
    """A compact tanh RNN for phrase-level speech recognition."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
        learning_rate: float = 0.005,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        scale = 0.08
        self.Wxh = rng.normal(0, scale, (input_size, hidden_size)).astype(np.float32)
        self.Whh = rng.normal(0, scale, (hidden_size, hidden_size)).astype(np.float32)
        self.bh = np.zeros(hidden_size, dtype=np.float32)
        self.Why = rng.normal(0, scale, (hidden_size, output_size)).astype(np.float32)
        self.by = np.zeros(output_size, dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h, _ = self._forward(x)
        representation = h.mean(axis=0)
        logits = representation @ self.Why + self.by
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x)))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 80) -> TrainingHistory:
        losses: list[float] = []
        accuracies: list[float] = []
        indices = np.arange(len(x_train))

        for epoch in range(1, epochs + 1):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            correct = 0

            for idx in indices:
                loss, prediction = self._train_one(x_train[idx], int(y_train[idx]))
                epoch_loss += loss
                correct += int(prediction == int(y_train[idx]))

            losses.append(epoch_loss / len(x_train))
            accuracies.append(correct / len(x_train))
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"epoch={epoch:03d} loss={losses[-1]:.4f} accuracy={accuracies[-1]:.2%}")

        return TrainingHistory(losses=losses, accuracies=accuracies)

    def save(self, path: str, labels: list[str]) -> None:
        np.savez(
            path,
            Wxh=self.Wxh,
            Whh=self.Whh,
            bh=self.bh,
            Why=self.Why,
            by=self.by,
            labels=np.array(labels),
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.learning_rate,
        )

    @classmethod
    def load(cls, path: str) -> tuple["SimpleRNNClassifier", list[str]]:
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            learning_rate=float(data["learning_rate"]),
        )
        model.Wxh = data["Wxh"]
        model.Whh = data["Whh"]
        model.bh = data["bh"]
        model.Why = data["Why"]
        model.by = data["by"]
        labels = [str(label) for label in data["labels"].tolist()]
        return model, labels

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros((x.shape[0], self.hidden_size), dtype=np.float32)
        previous = np.zeros(self.hidden_size, dtype=np.float32)
        for t in range(x.shape[0]):
            previous = np.tanh(x[t] @ self.Wxh + previous @ self.Whh + self.bh)
            h[t] = previous
        return h, previous

    def _train_one(self, x: np.ndarray, target: int) -> tuple[float, int]:
        h, _ = self._forward(x)
        representation = h.mean(axis=0)
        logits = representation @ self.Why + self.by
        probabilities = _softmax(logits)
        loss = -np.log(probabilities[target] + 1e-9)
        prediction = int(np.argmax(probabilities))

        dlogits = probabilities
        dlogits[target] -= 1.0

        dWhy = np.outer(representation, dlogits)
        dby = dlogits
        dh_next = np.zeros(self.hidden_size, dtype=np.float32)
        dh_from_output = (dlogits @ self.Why.T) / x.shape[0]

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(x.shape[0])):
            dh = dh_from_output + dh_next
            dtanh = dh * (1.0 - h[t] ** 2)
            dbh += dtanh
            dWxh += np.outer(x[t], dtanh)
            previous_h = h[t - 1] if t > 0 else np.zeros(self.hidden_size, dtype=np.float32)
            dWhh += np.outer(previous_h, dtanh)
            dh_next = dtanh @ self.Whh.T

        for gradient in (dWxh, dWhh, dbh, dWhy, dby):
            np.clip(gradient, -3.0, 3.0, out=gradient)

        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.bh -= self.learning_rate * dbh
        self.Why -= self.learning_rate * dWhy
        self.by -= self.learning_rate * dby

        return float(loss), prediction


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
