import os
import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Callable
import tqdm

class VlasovPoissonSolver:
    def __init__(self, nx: int, nv: int, nt: int, x_range: tuple, v_range: tuple, t_range: tuple, device: str):
        self.nx = nx
        self.nv = nv
        self.nt = nt
        self.x_range = x_range
        self.v_range = v_range
        self.t_range = t_range
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_checkpoint_path = None
        print("Instance of VlasovPoissonSolver created.")

    def make_grid(self):
        x = torch.linspace(self.x_range[0], self.x_range[1], self.nx).reshape(-1, 1)
        v = torch.linspace(self.v_range[0], self.v_range[1], self.nv).reshape(-1, 1)
        t = torch.linspace(self.t_range[0], self.t_range[1], self.nt).reshape(-1, 1)

        X, V, T = torch.meshgrid(x.squeeze(), v.squeeze(), t.squeeze(), indexing="ij")

        X = X.requires_grad_(True)
        V = V.requires_grad_(True)
        T = T.requires_grad_(True)

        self.X = X.to(self.device)
        self.V = V.to(self.device)
        self.T = T.to(self.device)
        return self.X, self.V, self.T

    def save_checkpoint(self, model, optimizer, loss, model_name, hyperparameters):
        self.checkpoint_dir = f"checkpoint_{model_name}_nx{self.nx}_nv{self.nv}_nt{self.nt}_epochs{hyperparameters['epochs']}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "model_checkpoint.pkl")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "hyperparameters": hyperparameters,
        }
        torch.save(checkpoint, checkpoint_path)
        self.model_checkpoint_path = checkpoint_path

    def train_step(self, model, loss_fn, optimizer, scheduler=None):
        # Forward Pass
        prediction = model(self.X, self.V, self.T, self.nx, self.nv, self.nt)

        # Compute Loss
        loss = loss_fn(model, self.X, self.V, self.T, self.nx, self.nv, self.nt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step(loss.item())

        return loss.item(), model.state_dict()

    def train(self, model, loss_fn, optimizer, epochs, model_name, get_loss_curve=False, scheduler=None):
        try:
          model = model.to(self.device)
          best_loss = float("inf")
          loss_values = []

          for epoch in tqdm.tqdm(range(epochs)):
              loss, model_params = self.train_step(model, loss_fn, optimizer, scheduler)
              loss_values.append(loss)

              if loss < best_loss:
                  best_loss = loss
                  self.save_checkpoint(model, optimizer, loss, model_name, {"epochs": epochs})

          print(f"Best Loss Achieved: {best_loss}")

          if get_loss_curve:
              plt.figure(figsize=(8, 5))
              plt.plot(range(1, epochs + 1), loss_values, marker="o", linestyle="-", color="r", label="Loss")
              plt.xlabel("Epoch")
              plt.ylabel("Loss")
              plt.title("Loss Curve Over Epochs")
              plt.legend()
              plt.grid()
              loss_curve_path = os.path.join(self.checkpoint_dir, "loss_curve.png")
              plt.savefig(loss_curve_path)
              print(f"Loss curve saved at {loss_curve_path}")
              plt.show()

              return self.model_checkpoint_path
        except AttributeError:
            print("Make grid first by calling make_grid()")

    def animate_final_prediction(self, model_class, model_checkpoint_path=None):
        try:
            if model_checkpoint_path is None:
                if self.model_checkpoint_path is None:
                    raise AttributeError("Model checkpoint not found. Train the model first.")
                model_checkpoint_path = self.model_checkpoint_path

            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            model = model_class().to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            with torch.no_grad():
                F_all = model(self.X, self.V, self.T, self.nx, self.nv, self.nt)  # Compute for all time steps

            def update(frame):
                ax.clear()
                ax.plot_surface(self.X.detach().cpu().numpy()[:,:,frame], self.V.detach().cpu().numpy()[:,:,frame], F_all.detach().cpu().numpy()[:,:,frame], cmap="viridis")
                ax.set_xlabel("X")
                ax.set_ylabel("V")
                ax.set_zlabel("F")
                ax.set_title(f"Predicted Distribution Function at Time {frame}")

            ani = FuncAnimation(fig, update, frames=self.nt, repeat=False)
            # Ensure checkpoint directory exists
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # Define file path and save animation
            animation_path = os.path.join(self.checkpoint_dir, "vlasov_distribution.gif")
            ani.save(animation_path, writer=PillowWriter(fps=10))

            # Store path in self.model_checkpoint_path
            self.model_checkpoint_path = animation_path
            print(f"Animation saved at: {animation_path}")

            # Close figure to free memory
            plt.close(fig)
            plt.show()
        except AttributeError as e:
            print(e)
