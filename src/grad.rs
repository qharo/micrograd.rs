use std::ops::{Add, Mul, Sub};
use std::rc::Rc;
use std::cell::RefCell;
use rand::Rng;


#[derive(Debug, Clone)]
enum Op {
    None,
    Add,
    Mul,
    Tanh
}

// param contains the values inside a node
// nodes need to be used by multiple 
#[derive(Debug, Clone)]
pub struct Node(Rc<RefCell<Param>>);

#[derive(Debug, Clone)]
struct Param {
    val: f64,
    grad: f64,
    children: Vec<Node>,
    op: Op,
}

impl Node {
    pub fn new(val: f64) -> Self {
        Node(Rc::new(RefCell::new(Param {
            val,
            grad: 0.0,
            children: Vec::new(),
            op: Op::None
        })))
    }

    pub fn val(&self) -> f64 {
        self.0.borrow().val
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn tanh(&self) -> Node {
        let result = Node::new(self.val().tanh());
        result.0.borrow_mut().children.push(self.clone());
        result.0.borrow_mut().op = Op::Tanh;
        result
    }

    pub fn square(&self) -> Node {
        self.clone() * self.clone()
    }

    pub fn backward_pass(&self) {
        // immutable borrow for getting grad
        let node = self.0.borrow();
        
        match node.op {
            Op::Add => {
                let grad = node.grad;
                drop(node);
                for child in &self.0.borrow().children {
                    let old_grad = child.grad();
                    child.set_grad(old_grad + grad); 
                    // mutable borrow for modifying children
                }
            }
            Op::Mul => {
                if self.0.borrow().children.len() == 2 {
                    let grad = node.grad;
                    let val0 = self.0.borrow().children[0].val();
                    let val1 = self.0.borrow().children[1].val();
                    drop(node);
                    
                    let old_grad0 = self.0.borrow().children[0].grad();
                    let old_grad1 = self.0.borrow().children[1].grad();
                    
                    self.0.borrow().children[0].set_grad(old_grad0 + val1 * grad);
                    self.0.borrow().children[1].set_grad(old_grad1 + val0 * grad);
                    
                }
            }
            Op::Tanh => {
                if let Some(child) = self.0.borrow().children.first() {
                    let val = self.val();
                    let der = 1.0 - val * val;
                    let grad = node.grad;
                    drop(node);
                    
                    let old_grad = child.grad();
                    child.set_grad(old_grad + der * grad);
                }
            }
            Op::None => {}
        }
        
        // Recursively apply to children
        for child in &self.0.borrow().children {
            child.backward_pass();
        }
    }
}

impl Add for Node {
    type Output = Node;

    fn add(self, other: Self) -> Self::Output {
        let result = Node::new(self.val() + other.val());
        result.0.borrow_mut().children.push(self);
        result.0.borrow_mut().children.push(other);
        result.0.borrow_mut().op = Op::Add;
        result
    }
}
impl Mul for Node {
    type Output = Node;

    fn mul(self, other: Self) -> Self::Output {
        let result = Node::new(self.val() * other.val());
        result.0.borrow_mut().children.push(self);
        result.0.borrow_mut().children.push(other);
        result.0.borrow_mut().op = Op::Mul;
        result
    }
}
impl Sub for Node {
    type Output = Node;

    fn sub(self, other: Self) -> Self::Output {
        self.clone() + other.clone()*Node::new(-1.0)
    }
}



#[derive(Debug, Clone)]
pub struct Neuron {
    n_in: i64,
    pub w: Vec<Node>,
    pub b: Node,
}

impl Neuron {
    pub fn new(n_in: i64) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize with smaller weights to prevent saturation
        let w = (0..n_in)
            .map(|_| Node::new(rng.gen_range(-0.1..0.1)))
            .collect();
            
        let b = Node::new(rng.gen_range(-0.1..0.1));
        
        Neuron { n_in, w, b }
    }

    pub fn forward(&self, x: Vec<Node>) -> Node {
        let mut act = self.b.clone();
        
        for i in 0..self.n_in as usize {
            let weight = self.w[i].clone();
            let input = x[i].clone();
            let weighted_input = weight * input;
            act = act + weighted_input;
        }
        
        act.tanh()
    }

    pub fn update_params(&self, learning_rate: f64) {
        // Add gradient clipping
        let clip_value = 1.0;
        
        for w in &self.w {
            let grad = w.grad().clamp(-clip_value, clip_value);
            let mut node = w.0.borrow_mut();
            node.val -= learning_rate * grad;
        }
        
        let grad = self.b.grad().clamp(-clip_value, clip_value);
        let mut b = self.b.0.borrow_mut();
        b.val -= learning_rate * grad;
    }

    pub fn zero_grad(&self) {
        for w in &self.w {
            w.set_grad(0.0);
        }
        self.b.set_grad(0.0);
    }

}

// ============= LAYER =============
#[derive(Debug, Clone)]
pub struct Layer{
    n_in: i64,
    n_out: i64,
    neurons: Vec<Neuron>
}
impl Layer {
    pub fn new(n_in: i64, n_out: i64) -> Layer{
        let mut neurons: Vec<Neuron> = Vec::new();
        for i in 1..=n_out {
            neurons.push(Neuron::new(n_in));
        }

        Layer{
            n_in: n_in,
            n_out: n_out,
            neurons: neurons
        }
    }

    pub fn forward(&mut self, x: Vec<Node>) -> Vec<Node> {
        let mut outputs: Vec<Node> = vec![];
        for i in 0..self.n_out as usize {
            outputs.push(self.neurons[i].forward(x.clone()));
        }
        outputs
    }    
    
    pub fn update_params(&mut self, step_size: f64) {
        for neuron in self.neurons.iter_mut(){
            neuron.update_params(step_size);
        }
    }

    pub fn zero_grad(&mut self) {
        for neuron in self.neurons.iter_mut(){
            neuron.zero_grad();
        }
    }
}


// ============= MLP =============
#[derive(Debug, Clone)]
pub struct MLP{
    n_in: i64,
    n_outs: Vec<i64>,
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(n_in: i64, n_outs: Vec<i64>) -> MLP{
        let mut layers: Vec<Layer> = vec![Layer::new(n_in, n_outs[0])];
        for i in 1..n_outs.len() {
            layers.push(Layer::new(n_outs[i-1], n_outs[i]));
        }

        MLP{
            n_in: n_in,
            n_outs: n_outs,
            layers: layers
        }
    }

    pub fn forward(&mut self, x: Vec<Node>) -> Vec<Node> {
        let mut outputs: Vec<Node> = x;
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(outputs);
        }
        outputs
    }

    pub fn update_params(&mut self, step_size: f64) {
        for layer in self.layers.iter_mut(){
            layer.update_params(step_size)
        }
    }

    pub fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut(){
            layer.zero_grad();
        }
    }
}
