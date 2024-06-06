import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as agnp
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation
from copy import deepcopy
from lazydl.nn import Module
from lazydl.optim import SGD, Adam, RMSProp, MSGD, AdaGrad

num_steps = 100


class Hole(Module):
    def __init__(self, x_init: float = 0.0, y_init: float = 0.0):
        super().__init__()
        self._parameters['x'] = np.array([x_init])
        self._parameters['y'] = np.array([y_init])
        self._gradients['x'] = np.zeros_like(self._parameters['x'])
        self._gradients['y'] = np.zeros_like(self._parameters['y'])

    def forward(self, params):
        x, y = params
        z = 0.0
        
        def __f2(x, y, x_mean, y_mean, x_sig, y_sig, depth):
            normalizing = 1 / (2 * agnp.pi * x_sig * y_sig)
            x_exp = (-1 * agnp.square(x - x_mean)) / (2 * agnp.square(x_sig))
            y_exp = (-1 * agnp.square(y - y_mean)) / (2 * agnp.square(y_sig))
            return -depth * normalizing * agnp.exp(x_exp + y_exp)
        
        z = __f2(x, y, x_mean=-.0, y_mean=-.0, x_sig=0.8, y_sig=0.8, depth=.5)
        return z

    def backward(self, grad_out):
        x = self._parameters['x']
        y = self._parameters['y']
        params = np.array([x[0], y[0]])
        gradients = grad(self.forward)(params)
        self._gradients['x'] = np.array([gradients[0]])
        self._gradients['y'] = np.array([gradients[1]])

        
def init_surface_fig():
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    model = Hole()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = model.forward(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.view_init(elev=0, azim=-135)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title('Different learning rate')

    return fig, ax


def get_result_path(optim):
    model = optim.model
    params = np.array([model._parameters['x'][0], model._parameters['y'][0]])
    path = [params.copy()]

    for _ in range(num_steps):
        grads = model.backward(params)
        optim.step()
        params = np.array([model._parameters['x'][0], model._parameters['y'][0]])
        path.append(params.copy())


    path = np.array(path)
    z_path = np.array([model.forward(p) for p in path])

    return path, z_path


def get_plot_animaiton(fig, ax, optim_config, paths, z_paths):
    lines = [ax.plot([], [], [], color=val[1], label=key)[0] for key, val in optim_config.items()]
    dots = [ax.plot([], [], [], marker='o', color=val[2])[0] for key, val in optim_config.items()]

    def init():
        for line, dot in zip(lines, dots):
            line.set_data([], [])
            line.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
        return lines + dots

    def animate(i):
        for j, (line, dot, path, Z_path) in enumerate(zip(lines, dots, paths, z_paths)):
            line.set_data(path[:i+1, 0], path[:i+1, 1])
            line.set_3d_properties(Z_path[:i+1])
            dot.set_data([path[i, 0]], [path[i, 1]])
            dot.set_3d_properties(Z_path[i])
        return lines + dots

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=10, blit=True, repeat=True)
    return ani


def main():
    ## modify here
    init_p = [.8, -.8]
    LR = .01
    model = Hole(x_init=init_p[0], y_init=init_p[1])

    optimizers = {
        'SGD (LR = 1e-2)': [SGD(deepcopy(model), lr=LR * 1), 'g', 'g'],
        'SGD (LR = 1e-1)': [SGD(deepcopy(model), lr=LR * 10), 'c', 'c'],
        'SGD (LR = 1)': [SGD(deepcopy(model), lr=LR * 100), 'r', 'r'],
        'SGD (LR = 10)': [SGD(deepcopy(model), lr=LR * 1000), 'm', 'm'],
        }

    fig, ax = init_surface_fig()
    paths, z_paths = [], []
    for optim in optimizers.values():
        path, z_path = get_result_path(optim[0])
        paths.append(path)
        z_paths.append(z_path)

    ani = get_plot_animaiton(fig, ax, optimizers, paths, z_paths)
    plt.legend()
    
    # ani.save('optimizers.gif', writer='ffmpeg', fps=30)

    plt.show()




if __name__ == '__main__':
    main()
