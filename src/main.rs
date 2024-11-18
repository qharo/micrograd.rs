mod grad;
use crate::grad::{Neuron, Layer, MLP, Node};
use rand::Rng;
use rand::prelude::SliceRandom;  // Added for shuffle
use rand::thread_rng;

fn main() {
    // Generate spiral dataset
    let n_points = 100;
    let noise = 0.1;
    let mut training_data = Vec::new();
    let mut rng = thread_rng();
    
    // Generate two spirals
    for i in 0..n_points {
        let r = i as f64 / n_points as f64;
        let t = i as f64 * 4.0;
        
        // First spiral (class 0)
        let x1 = r * (t).cos() + rng.gen_range(-noise..noise);
        let y1 = r * (t).sin() + rng.gen_range(-noise..noise);
        training_data.push((vec![x1, y1], vec![0.0]));
        
        // Second spiral (class 1)
        let x2 = r * (t + std::f64::consts::PI).cos() + rng.gen_range(-noise..noise);
        let y2 = r * (t + std::f64::consts::PI).sin() + rng.gen_range(-noise..noise);
        training_data.push((vec![x2, y2], vec![1.0]));
    }
    
    // Deeper network: 2 -> 32 -> 32 -> 16 -> 8 -> 1
    let mut mlp = MLP::new(2, vec![16, 8, 1]);
    
    // Adjusted training parameters
    let initial_learning_rate = 0.03;
    let epochs = 200;
    
    // Training loop
    for epoch in 0..epochs {
        // Learning rate decay
        let learning_rate = initial_learning_rate / (1.0 + epoch as f64 * 0.001);
        
        let mut total_loss = 0.0;
        
        // Shuffle training data
        let mut indices: Vec<usize> = (0..training_data.len()).collect();
        indices.shuffle(&mut rng);
        
        for &idx in indices.iter() {
            let (inputs, targets) = &training_data[idx];
            
            // Forward pass
            let x: Vec<Node> = inputs.iter()
                .map(|&val| Node::new(val))
                .collect();
                
            let outputs = mlp.forward(x);
            let expected = Node::new(targets[0]);
            let diff = outputs[0].clone() - expected;
            let loss = diff.square();
            
            total_loss += loss.val();
            loss.set_grad(1.0);
            
            if epoch % 1 == 0 && idx < 4 {
                println!(
                    "Epoch {}, Point ({:.3}, {:.3}), Target: {}, Output: {:.4}, Loss: {:.4}",
                    epoch, inputs[0], inputs[1], targets[0], outputs[0].val(), loss.val()
                );
            }
            
            loss.backward_pass();
            mlp.update_params(learning_rate);
            mlp.zero_grad();
        }
        
        if epoch % 1 == 0 {
            println!("Epoch {}: Average loss = {:.4} (lr = {:.4})", 
                    epoch, total_loss / (2.0 * n_points as f64), learning_rate);
            println!("");
        }
        
        // Early stopping if loss is good enough
        if total_loss / (2.0 * n_points as f64) < 0.01 {
            println!("Reached target loss at epoch {}", epoch);
            break;
        }
    }
    
    // Test grid points to visualize decision boundary
    println!("\nDecision Boundary Sample:");
    let grid_points = [-1.0, -0.5, 0.0, 0.5, 1.0];
    for &y in grid_points.iter().rev() {
        let mut line = String::new();
        for &x in grid_points.iter() {
            let x: Vec<Node> = vec![Node::new(x), Node::new(y)];
            let output = mlp.forward(x)[0].val();
            let symbol = if output > 0.5 { "1" } else { "0" };
            line.push_str(&format!("{} ", symbol));
        }
        println!("{}", line);
    }
    
    // Test accuracy on training data
    let mut correct = 0;
    let mut class0_correct = 0;
    let mut class1_correct = 0;
    let mut class0_total = 0;
    let mut class1_total = 0;
    
    for (inputs, targets) in training_data.iter() {
        let x: Vec<Node> = inputs.iter()
            .map(|&val| Node::new(val))
            .collect();
            
        let output = mlp.forward(x)[0].val();
        let predicted = if output > 0.5 { 1.0 } else { 0.0 };
        
        if (predicted - targets[0]).abs() < 1e-5 {
            correct += 1;
            if targets[0] < 0.5 { class0_correct += 1; }
            else { class1_correct += 1; }
        }
        
        if targets[0] < 0.5 { class0_total += 1; }
        else { class1_total += 1; }
    }
    
    println!("\nFinal Results:");
    println!("Overall accuracy: {:.2}%", 100.0 * correct as f64 / (2.0 * n_points as f64));
    println!("Class 0 accuracy: {:.2}%", 100.0 * class0_correct as f64 / class0_total as f64);
    println!("Class 1 accuracy: {:.2}%", 100.0 * class1_correct as f64 / class1_total as f64);
}