# This file includes code originally from the lietorch repository:
# https://github.com/princeton-vl/lietorch
# Licensed under the BSD-3 License. See THIRD_PARTY_LICENSES.md for details.

import torch
import torch.nn.functional as F

from vipe.ext import lietorch_ext as lietorch_backends


class GroupOp(torch.autograd.Function):
    """group operation base class"""

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        # CPU fallback: the lietorch CUDA kernels (Inv/Exp/Log/Mul/Adj/Act/...)
        # access-violate on sm_120 (RTX 5090) even after rebuilding for that
        # arch. The CPU kernels (compiled from lietorch_cpu.cpp) work fine;
        # route through them and copy the result back to the original device.
        # SE3/SO3/etc tensors are small (~B*7 elements) so the GPU<->CPU
        # roundtrip is sub-millisecond.
        device = inputs[0].device
        if device.type == "cuda":
            cpu_inputs = tuple(x.detach().cpu().contiguous() for x in inputs)
            out_cpu = cls.forward_op(ctx.group_id, *cpu_inputs)
            return out_cpu.to(device)
        return cls.forward_op(ctx.group_id, *inputs)

    @classmethod
    def backward(cls, ctx, grad):
        error_str = "Backward operation not implemented for {}".format(cls)
        assert cls.backward_op is not None, error_str

        inputs = ctx.saved_tensors
        grad = grad.contiguous()
        # Same CPU fallback for the backward kernel.
        device = grad.device
        if device.type == "cuda":
            cpu_grad = grad.detach().cpu()
            cpu_inputs = tuple(x.detach().cpu().contiguous() for x in inputs)
            grad_inputs = cls.backward_op(ctx.group_id, cpu_grad, *cpu_inputs)
            grad_inputs = tuple(g.to(device) for g in grad_inputs)
        else:
            grad_inputs = cls.backward_op(ctx.group_id, grad, *inputs)
        return (None,) + tuple(grad_inputs)


class Exp(GroupOp):
    """exponential map"""

    forward_op, backward_op = lietorch_backends.expm, lietorch_backends.expm_backward


class Log(GroupOp):
    """logarithm map"""

    forward_op, backward_op = lietorch_backends.logm, lietorch_backends.logm_backward


class Inv(GroupOp):
    """group inverse"""

    forward_op, backward_op = lietorch_backends.inv, lietorch_backends.inv_backward


class Mul(GroupOp):
    """group multiplication"""

    forward_op, backward_op = lietorch_backends.mul, lietorch_backends.mul_backward


class Adj(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = lietorch_backends.adj, lietorch_backends.adj_backward


class AdjT(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = lietorch_backends.adjT, lietorch_backends.adjT_backward


class Act3(GroupOp):
    """action on point"""

    forward_op, backward_op = lietorch_backends.act, lietorch_backends.act_backward


class Act4(GroupOp):
    """action on point"""

    forward_op, backward_op = lietorch_backends.act4, lietorch_backends.act4_backward


class Jinv(GroupOp):
    """adjoint operator"""

    forward_op, backward_op = lietorch_backends.Jinv, None


class ToMatrix(GroupOp):
    """convert to matrix representation"""

    forward_op, backward_op = lietorch_backends.as_matrix, None


### conversion operations to/from Euclidean embeddings ###


class FromVec(torch.autograd.Function):
    """convert vector into group object"""

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), torch.linalg.pinv(J)).squeeze(-2)


class ToVec(torch.autograd.Function):
    """convert group object to vector"""

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), J).squeeze(-2)
