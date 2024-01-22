use crate::{
    grads::Gradients,
    graph::{
        checkpoint::NodeStates,
        NodeRef, Requirement, {Graph, Step},
    },
    tensor::AutodiffTensor,
};
use burn_tensor::{backend::Backend, ops, Shape};
use std::{marker::PhantomData, process::Output};

use super::OpsSpec;

/// Operation in preparation.
///
/// There are 3 different modes: 'Init', 'Tracked' and 'UnTracked'.
/// Each mode has its own set of functions to minimize cloning for unused backward states.
#[derive(new)]
pub struct OpsPrep<OS, B, I, O, const D: usize, const N: usize, Mode = Init> {
    nodes: [NodeRef; N],
    graphs: [Graph; N],
    requirement: Requirement,
    ops_spec: OS,
    phantom_backend: PhantomData<B>,
    phantom_input: PhantomData<I>,
    phantom_output: PhantomData<O>,
    marker: PhantomData<Mode>,
}

/// Init operation tag.
pub struct Init;
/// Tracked operation tag.
pub struct Tracked;
/// Untracked operation tag.
pub struct UnTracked;

impl<OS, B, const D: usize, const N: usize> OpsPrep<OS, B, (), (), D, N, Init>
where
    B: Backend,
    OS: OpsSpec<B, D, N, Input = (), Output = ()>,
{
    /// Prepare a stateless operation.
    pub fn stateless(self, output: <B as Backend>::TensorPrimitive<D>) -> AutodiffTensor<B, D> {
        match self.stateful() {
            OpsKind::Tracked(prep) => prep.finish(None, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

impl<OS, B, const D: usize, const N: usize, I, O> OpsPrep<OS, B, I, O, D, N, Init>
where
    B: Backend,
    OS: OpsSpec<B, D, N, Input = I, Output = O>,
    I: Clone + Send + Sync + std::fmt::Debug + 'static,
    O: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Prepare an operation that requires a state during the backward pass.
    pub fn stateful(self) -> OpsKind<OS, B, I, O, D, N> {
        match self.requirement.is_none() {
            false => OpsKind::Tracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.ops_spec,
            )),
            true => OpsKind::UnTracked(OpsPrep::new(
                self.nodes,
                self.graphs,
                self.requirement,
                self.ops_spec,
            )),
        }
    }
}

impl<OS, B, const D: usize, const N: usize, I, O> OpsPrep<OS, B, I, O, D, N, UnTracked>
where
    B: Backend,
    OS: OpsSpec<B, D, N, Input = I, Output = O>,
    I: Clone + Send + Sync + std::fmt::Debug + 'static,
    O: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Finish the preparation of an untracked operation and returns the output tensor.
    pub fn finish(self, output: <B as Backend>::TensorPrimitive<D>) -> AutodiffTensor<B, D> {
        AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
        )
    }
}

impl<OS, B, const D: usize, const N: usize, I, O> OpsPrep<OS, B, I, O, D, N, Tracked>
where
    B: Backend,
    OS: OpsSpec<B, D, N, Input = I, Output = O>,
    I: Clone + Send + Sync + std::fmt::Debug + 'static,
    O: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    /// Finish the preparation of a tracked operation and returns the output tensor.
    pub fn finish(
        self,
        // state: S, // was given to ops but i removed it
        ops_spec: Option<OS>,
        output: <B as Backend>::TensorPrimitive<D>,
    ) -> AutodiffTensor<B, D> {
        let autodiff_tensor = AutodiffTensor::from_parents(
            output,
            &self.nodes,
            self.graphs.into_iter(),
            self.requirement,
        );
        let parents = self.nodes.map(|node| node.clone_if_require_grad());
        let ops = Ops::new(parents, autodiff_tensor.node.clone());
        // probably should have ops_spec in order to recompute stuff
        // let ops = Ops::new(parents, autodiff_tensor.node.clone(), ops_spec);

        // Access autodiff_tensor.graph, ask it to register
        // <nodeid, output>, nodeid inside tensor
        match ops_spec {
            Some(ops_spec) => match ops_spec.bottleneck() {
                ComputeBound => {
                    autodiff_tensor.register_output(output);
                }
                MemoryBound => {}
            },
            None => {}
        }

        autodiff_tensor.register_step(OpsStep::new(ops, self.ops_spec))
    }
}

/// Enum used before finishing tracked and untracked operations.
pub enum OpsKind<OS, B, I, O, const D: usize, const N: usize> {
    /// Tracked operation preparation.
    Tracked(OpsPrep<OS, B, I, O, D, N, Tracked>),
    /// Untracked operation preparation.
    UnTracked(OpsPrep<OS, B, I, O, D, N, UnTracked>),
}

/// Operation containing its parent nodes, its own node and the backward step state.
#[derive(new, Debug)]
pub struct Ops<I, O, const N: usize> {
    /// Parents nodes.
    pub parents: [Option<NodeRef>; N],
    /// The node.
    pub node: NodeRef,
    pub _input: PhantomData<I>,
    pub _output: PhantomData<O>,
}

impl<I, O, const N: usize> Ops<I, O, N> {
    pub fn fetch_inputs(&self, states: &NodeStates) -> I {
        states.get_input(self.node)
    }

    pub(crate) fn forward(&self, inputs: Vec<Box<dyn State>>) -> dyn State {
        self.ops_spec.forward(inputs)
    }
}

/// Operation implementing backward [step](Step) with type erasing.
#[derive(new, Debug)]
struct OpsStep<B, T, I, O, const D: usize, const N: usize>
where
    B: Backend,
    T: OpsSpec<B, D, N, Input = I, Output = O>,
    I: Clone + Send + Sync + std::fmt::Debug + 'static,
    O: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    ops: Ops<I, O, N>,
    backward: T,
    phantom: PhantomData<B>,
}

impl<B, T, I, O, const D: usize, const N: usize> Step for OpsStep<B, T, I, O, D, N>
where
    B: Backend,
    T: OpsSpec<B, D, N, Input = I, Output = O>,
    I: Clone + Send + Sync + std::fmt::Debug + 'static,
    O: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    fn step(self: Box<Self>, grads: &mut Gradients, states: &NodeStates) {
        self.backward.backward(self.ops, grads, states);
    }

    fn node(&self) -> NodeRef {
        self.ops.node.clone()
    }
}

/// Make sure the grad tensor has the given shape.
///
/// If broadcasting happened during the forward pass, the gradients will be sum along the
/// broadcasted dimension.
pub fn broadcast_shape<B: Backend, const D: usize>(
    mut grad: B::TensorPrimitive<D>,
    shape: &Shape<D>,
) -> B::TensorPrimitive<D> {
    let shape_grad = B::shape(&grad);

    for i in 0..D {
        if shape_grad.dims[i] != shape.dims[i] {
            if shape.dims[i] != 1 {
                panic!(
                    "Invalid broadcast shapes: Next grad shape {:?}, Previous grad shape {:?}. {}",
                    shape.dims, shape_grad.dims, "Expected the shape of the next grad to be 1."
                );
            }
            grad = B::sum_dim(grad, i);
        }
    }

    grad
}
