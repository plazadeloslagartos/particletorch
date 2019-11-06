"""
Boostrap particle filter (Linear Dynamic Model) implemented with PyTorch.
Allows for batching over a whole dataset during forward filtering and backward smoothing.
Provides an expectation maximization routine for optimization of parameters via PyTorch autograd functionality.
"""


import torch
from torch import Tensor
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
from torch.nn import Parameter
import numpy as np


class BootstrapFilter:
    def __init__(self, x_init=Tensor([0.0]), trans_mat=Tensor([1]), obs_mat=Tensor([1]), ctrl_mat=None,
                 proposal=mvn, likelihood=mvn, p_params=(0.1,), l_params=(0.5,), num_particles=1000,
             n_eff_thresh=0.5, track_grad=False, param_guides={}):
        """
        :param x_init: Tensor, Initial state Tensor
        :param trans_mat: Tensor, Transition Matrix
        :param obs_mat: Tensor, Observation (extraction) Matrix
        :param ctrl_mat: Tensor, Control Matrix
        :param proposal: PyTorch Distribution class, Proposal Distribution (Transition Noise Model)
        :param likelihood: PyTorch Distribution class, Likelihood Distribution (Measurement Noise Model)
        :param p_params: Tensor, parameters for Proposal Distribution Constructor
        :param l_params: Tensor, parameters for Likelihood Distribution Constructor
        :param num_particles: int, Number of particles for filter
        :param n_eff_thresh: int, number of effective particles threshold to trigger resampling
        :param track_grad: bool, whether or not to track parameter gradients
        :param param_guides: dict, dictionary of transformation functions for reshaping/manipulating attributes to avoid
            unnecessary/unwanted gradient tracking.  The key match the attribute name
            i.e., {'p_params': torch.diag} where p_params is a Tensor of size 2 will transform p_params into a diagonal
            2x2 matrix, yet gradients will be tracked only for the original values in p1.
        """
        self.proposal = proposal
        self.likelihood = likelihood
        self.p_params = p_params
        self.l_params = l_params
        self.trans_mat = trans_mat
        self.ctrl_mat = ctrl_mat
        self.obs_mat = obs_mat
        self.reset = 0
        self.num_particles = num_particles
        self.n_eff_thresh = n_eff_thresh
        self.param_guides = param_guides
        if x_init.size()[0] == 1:
            self.x = x_init.repeat(num_particles)
        else:
            self.x = x_init.repeat(num_particles, 1)
        self.w = torch.ones(num_particles, 1) / num_particles
        self.x_pred = None
        self.mu = None
        self.stdev = None
        self.track_grad = track_grad

    def get_param(self, param_name):
        """Gets attribute parameter and transforms it via any guides specified in self.param_guides"""
        noop = lambda x: x
        return self.param_guides.get(param_name, noop)(self.__getattribute__(param_name))

    def propagate_state(self, x, u=None):
        """
        Propagates State X forward using transition and control models
        :param x: Tensor, current state
        :param u: Tensor, control input
        :return: Tensor, progagated state
        """
        t_mat = self.get_param('trans_mat')
        c_mat = self.get_param('ctrl_mat')
        if t_mat.dim() == 1:
            t_mat = t_mat.unsqueeze(0)
        if c_mat is not None:
            if c_mat.dim() == 1:
                c_mat = c_mat.unsqueeze(0)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        prop_out = x @ t_mat.transpose(0, 1)
        if (c_mat is not None) and (u is not None):
            prop_out += u @ c_mat.transpose(0, 1)
        return prop_out

    def predict_obs(self, x):
        """
        Given propagated state X, predicts the observation given the Extraction Model
        :param x: Tensor, propagated state
        :return: Tensor, predicted observation
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        o_mat = self.get_param('obs_mat')
        if o_mat.dim() == 1:
            o_mat = o_mat.unsqueeze(0)
        return x @ o_mat.transpose(0, 1)

    def get_trans_prob(self, x, x_mu, u=None):
        """
        Given new observation of state estimate and current state estimate and control input, get transition probability
        :param x: Tensor, new observation of state estimate samples used to estimate state (particles)
        :param x_mu: Tensor, current state estimate samples used to estimate state (particles)
        :param u: Tensor, control input
        :return: Tensor, log transition probability
        """
        cov_mat = self.get_param('p_params')
        mu = self.propagate_state(x_mu, u)
        trans_dist = self.proposal(mu, cov_mat)
        output = trans_dist.log_prob(x)
        return output

    def get_obs_likelihood(self, y, x_mu):
        """
        Given new observation and previous state, calculate log-likelihood of new observation
        :param y: Tensor, observation
        :param x_mu: Tensor, current state samples used to estimate state (particles)
        :return:
        """
        cov_mat = self.get_param('l_params')
        mu = self.predict_obs(x_mu)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if mu.shape[-1] != len(cov_mat):
            mu = mu.unsqueeze(-1)
        if cov_mat.dim() == 1:
            cov_mat = cov_mat.unsqueeze(1)
        obs_dist = self.likelihood(mu, cov_mat)
        output = obs_dist.log_prob(y)
        return output.unsqueeze(output.dim())

    def draw_samples(self, u_new):
        """
        Draw samples from proposal model, including any control inputs
        :param u_new: Tensor, control inputs
        :return: Tensor, samples of state
        """
        x_new = self.propagate_state(self.x, u=u_new)
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(-1, 1)
        x_dist = self.proposal(x_new, torch.diag(self.p_params))
        self.x = torch.squeeze(x_dist.sample([1]))

    def get_n_effective_particles(self):
        """Calculate number of effective particles"""
        return 1 / (self.w ** 2).sum()

    def update_weights(self, y):
        """
        Given a new observation, update particle weights
        :param y: Tensor, new observation
        :return: Tensor, new weights
        """
        self.w *= torch.exp(self.get_obs_likelihood(y, self.x))
        self.w /= self.w.sum()

    def resample_weights(self):
        """
        Performs resampling of weights
        :return:
        """
        idxs = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=np.squeeze(self.w.numpy()))
        self.x = self.x[idxs]
        self.w = torch.ones_like(self.w) / self.num_particles

    def update_stats(self):
        """
        Calculates Mean and Std. Dev of current state estimate, assigns to attributes
        :return: None
        """
        self.mu = self.w.transpose(0, 1) @ self.x
        self.stdev = torch.sqrt(self.w.transpose(0, 1) @ (self.x - self.mu)**2)

    def update(self, y_new, u_new=None):
        """
        Update filter with new estimate
        :param y_new: Tensor, new observation
        :param u_new: Tensor, control input
        :return: None
        """
        if not self.track_grad:
            with torch.no_grad():
                self.update_(y_new, u_new)
        else:
            self.update_(y_new, u_new)

    def update_(self, y_new, u_new=None):
        """Utility function called by update()"""
        if np.isnan(y_new).any():
            self.x = self.propagate_state(self.x, u=u_new)
            self.reset = 0
        else:
            self.draw_samples(u_new)
            self.update_weights(y_new)
            self.update_stats()
            self.x_pred = self.x
            n_eff = self.get_n_effective_particles()
            if n_eff < self.num_particles * self.n_eff_thresh:
                self.resample_weights()
                self.reset = 1
            else:
                self.reset = 0


def forward_filter(pf, data, indicator=None):
    """
    Given a particle filter and Tensor of data and Indicators (control inputs), perform forward Bayesian filtering.
    All computations computed in one batch pass.
    :param pf: instance of particle filter
    :param data: Tensor, data over which to iterate
    :param indicator: Tensor, indicator functions (control inputs)
    :return: dict of Tensors, mu,
                              stdev,
                              weights
                              particles
                              resets (occurence of resets)
    """
    mu = []
    stdev = []
    particles = []
    weights = []
    resets = []

    if indicator is None:
        indicator = [None for idx in range(len(data))]

    for val, ind_val in zip(data, indicator):
        pf.update(val, u_new=ind_val)
        weights.append(pf.w)
        particles.append(pf.x_pred)
        mu.append(pf.mu)
        stdev.append(pf.stdev)
        resets.append(pf.reset)

    output = {
        'mu': torch.squeeze(torch.stack(mu)),
        'stdev': torch.squeeze(torch.stack(stdev)),
        'weights': torch.stack(weights),
        'particles': torch.stack(particles),
        #'conf': torch.squeeze(norm(mu, stdev).interval(0.95)).T,
        'resets': Tensor(resets)
    }
    return output


def backward_smoothing(pf, data, indicator=None):
    """
    Given an instance of a particle filter, dataset, and Indicators, performs Bayesian forward filtering followed by
    a backward smoothing pass.
    :param pf: particle filter instance
    :param data: Tensor, dataset over which to iterate
    :param indicator: Tensor, indicators
    :return: dict of Tensors, same as above but with "smoothed_weights" added
    """
    if indicator is None:
        indicator = [None for idx in range(len(data))]
    """Forward Filter followed by Backward Smoothing"""
    with torch.no_grad():
        forward_output = forward_filter(pf, data, indicator)
        weights = forward_output['weights']
        particles = forward_output['particles']
        x_tp1 = particles[-1]
        w_tp1 = torch.ones(x_tp1.shape[0], 1)/x_tp1.shape[0]
        smooth_weights = [w_tp1]

        for idx in range(weights.shape[0] - 2, -1, -1):
            w_t = weights[idx]
            x_t = particles[idx]
            if x_tp1.dim() == 1:
                x_eval = x_tp1.reshape((x_tp1.shape[0], 1, 1))
            else:
                x_eval = x_tp1.unsqueeze(1)
            num = torch.exp(pf.get_trans_prob(x_eval, x_t, u=indicator[idx])).transpose(0, 1)
            denom = (w_t.reshape(-1, 1) * num).sum(dim=0).reshape(-1, 1)
            new_weights = w_t * ((w_tp1 / denom).reshape(1, -1) * num).sum(dim=1).reshape(-1, 1)
            x_tp1 = x_t
            w_tp1 = new_weights
            smooth_weights.append(new_weights)
        res_idxs = np.where(forward_output['resets'])[0]
        out_weights = torch.stack(smooth_weights).flip(0)
        # Fix irregularities due to resampling
        for idx in res_idxs:
            pf.w = out_weights[idx - 1]
            pf.x = particles[idx - 1]
            y = data[idx]
            pf.update_weights(y)
            out_weights[idx] = pf.w

        forward_output.update({"smoothed_weights": out_weights})
    return forward_output


def backward_simulate(pf, data, num_trajectories, indicator=None):
    """
    Given an instance of a particle filter, dataset, and Indicators, performs Bayesian forward filtering followed by
    a backward smoothing via simulation pass.  This creates a new set of particle trajectories without path degeneration
    due to resampling.
    :param pf: particle filter instance
    :param data: Tensor, dataset over which to iterate
    :param num_trajectories: int, number of particle trajectories for backward simulation
    :param indicator: Tensor, indicators
    :return: dict of Tensors, same as above but with "smoothed_weights" added, "particles" is replaced with simulated
            trajectories
    """
    assert(num_trajectories < pf.num_particles)
    if indicator is None:
        indicator = [None for idx in range(len(data))]
    with torch.no_grad():
        def sampler(x, count, w_arr):
            idxs = np.random.choice(range(len(x)), size=count, p=np.squeeze(w_arr.detach().numpy()))
            return x[idxs]
        forward_output = forward_filter(pf, data, indicator=indicator)
        weights = forward_output['weights']
        particles = forward_output['particles']
        # Initialize M trajectories from final weights and particles
        x_m = sampler(particles[-1], num_trajectories, weights[-1])
        trajectories = [x_m]
        traj_weights = []

        # Iterate through historical data to flesh out trajectories
        for idx in range(weights.shape[0] - 2, -1, -1):
            w = weights[idx]
            p = particles[idx]
            x_tilde = trajectories[-1]
            if x_tilde.dim() == 1:
                x_tilde = x_tilde.unsqueeze(-1)
            if p.dim() == 1:
                p = p.unsqueeze(-1)
            new_weights = w * torch.exp(pf.get_trans_prob(x_tilde.unsqueeze(1), p, u=indicator[idx+1])).t()
            new_weights /= new_weights.sum(dim=0).reshape(1, -1)
            sampler_l = lambda w_arr: torch.squeeze(sampler(p, 1, w_arr))
            x_m = torch.stack([sampler_l(w_arr) for w_arr in new_weights.t()])
            trajectories.append(x_m)
        trajectories = torch.stack(trajectories).flip(0)

        # Recalculate weights
        for idx in range(weights.shape[0]):
            pf.w = torch.ones(num_trajectories, 1)
            pf.x = trajectories[idx]
            if pf.x.dim() == 1:
                pf.x = pf.x.unsqueeze(-1)
            pf.update_weights(data[idx])
            traj_weights.append(pf.w)

        # Update output
        traj_weights = torch.stack(traj_weights).flip(0)
        forward_output.update({'particles': trajectories})
        forward_output.update({'smoothed_weights': traj_weights})

    return forward_output


def get_filtered_output(particles, weights):
    """Given Tensors of particles and weights over time, returns the estimated expected state"""
    return np.stack([w @ p for w, p in zip(weights, particles)])


def em_qfunc(pf, data, weights, particles, indicator=None):
    """
    Evaluates Expectation Maximization 'Q' function (negative log prob)
    pf: instance of particle filter
    data: Tensor, dataset overwhich to iterate
    weights: Tensor, resultant weights from a forward or forward/backward pass
    particles: Tensor, resultant particles from a forward or forward/backward pass
    indicator: Tensor, indicator functions (control inputs)
    """
    def eval_prob(d, w, p, p_tm1, ind, trans=True, obs=True):
        ll = 0
        if trans:
            if p.dim() == 2:
                ll += (pf.get_trans_prob(p.unsqueeze(2), p_tm1.unsqueeze(2), u=ind).unsqueeze(-1) * w).sum()
            else:
                ll += (pf.get_trans_prob(p, p_tm1, u=ind) * torch.squeeze(w)).sum()

        if obs:
            if p.dim() == 2:
                ll += (pf.get_obs_likelihood(d, p.unsqueeze(-1)) * w).sum()
            else:
                ll += (pf.get_obs_likelihood(d, p) * w).sum()
        return ll

    invalid_idxs = np.hstack([[0], np.where(np.isnan(data))[0]])
    valid_idxs = np.setdiff1d(np.arange(len(data)), invalid_idxs)

    # Get transition likelihoods for all states and points
    if data.dim() == 1:
        data = data.unsqueeze(1).unsqueeze(1)
    if indicator is None:
        ind_val = indicator
    else:
        ind_val = indicator[valid_idxs].unsqueeze(1)
    Q = eval_prob(data[valid_idxs], weights[valid_idxs], particles[valid_idxs], particles[valid_idxs - 1],
                  ind_val)
    return -1.0 * Q


def pytorch_optimize(data, forward_func, fixed_params, optim_params, param_guides={}, max_epochs=120,
                     mode='backward_smooth', indicator=None, em_tol=1e-3, l_rate=1e-4, iter_tol=3.0):
    """
    Makes use of autograd to perform optimization of particle filter model parameters via expectation maximiation.
    :param data: Tensor, dataset over which to iterate
    :param forward_func: function used to establish "neural network forward pass" in order to track gradients relevant
                        to loss (i.e. EM Q function)
    :param fixed_params: dict of model params to not be optimized
    :param optim_params: dict of model params to be optimized (should be created with torch.nn.Parameter)
    :param param_guides: dict of param guides
    :param max_epochs: (int), max number of processing epochs
    :param mode: (str), particle filter backward smoothing method "backward_smooth" or "backward_simulate"
    :param indicator: Tensor, Indicators (control inputs)
    :param em_tol: float, tolerance for expectation maximization step
    :param l_rate: float, learning rate
    :param iter_tol: float, tolerance for termination between successive passes of EM routine
    :return: tuple final_params, tracked loss, processed output of final params
    """
    if indicator is None:
        indicator = [None for idx in range(len(data))]
    optimizer = optim.Adam(list(optim_params.values()),
                           lr=l_rate)
    full_params = fixed_params.copy()
    full_params.update(optim_params)
    full_params.update({'param_guides': param_guides})
    full_params['track_grad'] = False

    losses = [0]
    iter = 0
    iter_loss_diff = 10000

    while abs(iter_loss_diff) > iter_tol:
        pf = BootstrapFilter(**full_params)
        epoch = 0
        loss_diff = 10000
        prev_loss = 0
        if mode == 'backward_smooth':
            output = backward_smoothing(pf, data, indicator)
        elif mode == 'backward_simulate':
            output = backward_simulate(pf, data, full_params['num_particles']//4, indicator)
        else:
            raise ValueError("Invalid smoothing mode specified: {:s}".format(mode))
        weights = output['smoothed_weights']
        particles = output['particles']
        while (abs(loss_diff) > em_tol) and (epoch < max_epochs):
            if (epoch < 20) and (iter < 2):
                optimizer.lr = .01
            elif iter < 5:
                optimizer.lr = 1e-3
            else:
                optimizer.lr = 1e-4
            loss = forward_func(pf, data, weights, particles, indicator=indicator)
            if epoch > 0:
                prev_loss = track_loss
            track_loss = loss.item()
            loss_diff = track_loss - prev_loss
            optimizer.zero_grad()
            print('Loss at iteration {:d}, epoch {:d}, {:.3f}, ldiff: {:.3f}, iter_diff: {:.3f}'.format(iter,
                                                            epoch, track_loss, loss_diff, iter_loss_diff))
            loss.backward()
            optimizer.step()
            pf.p_params.data.clamp_(1e-4, 1.5)
            epoch += 1

        iter_loss_diff = abs(track_loss - losses[-1])
        losses.append(track_loss)
        iter += 1
    return full_params, losses[1:], data