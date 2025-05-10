use crate::linear_algebra::matrices::{self, multiply_matrices};
use super::utils::*; 
use std::iter::zip;

pub struct ExponentialRegression {
    c: f32, 
    k: f32
} 


impl ExponentialRegression {
    pub fn new() -> Self {
        return Self {
            c: 0.0,  
            k: 0.0
        }
    }

    pub fn fit(&mut self, data: &Vec<f32>, labels: &Vec<f32>, iterations: i32, step_size: f32) {
        // 1st parameter - k (exponent)
        // 2nd parameter - c (scalar)
        // y = Ce^-kx 
        let mut parameters = vec![1 as f32, 1 as f32];  
        for _ in 0..iterations {
            let [k, c] = [parameters[0], parameters[1]];
            let predictions: Vec<f32> = data.iter().map(|x_value| exp_x(*x_value, c, k)).collect(); 
            let residuals: Vec<Vec<f32>> = zip(labels, predictions.iter()).map(|(label, prediction)| vec![*label - *prediction]).collect(); 
            let residual_p_k: Vec<f32> = data.iter().map(|x_value| residual_p_k(*x_value, c, k)).collect();  
            let residual_p_c: Vec<f32> = data.iter().map(|x_value| residual_p_c(*x_value, k)).collect();  
            let jacobian_transposed = vec![residual_p_k, residual_p_c];
            let jacobian = matrices::transpose_matrix(&jacobian_transposed);
            let jacobian_dotted = matrices::multiply_matrices(&jacobian_transposed, &jacobian); 
            let jacobian_dotted_inverse = matrices::inverse_matrix(&jacobian_dotted);
            let residual_dotted = matrices::multiply_matrices(&jacobian_transposed, &residuals);
            let gradient = matrices::multiply_matrices(&jacobian_dotted_inverse, &residual_dotted); 
            parameters = zip(parameters, gradient).map(|(parameter, gradient_value)| {
                let gradient_value = gradient_value[0];
                parameter - (gradient_value * step_size) 
            }).collect();
        }
        self.k = parameters[0];
        self.c = parameters[1]; 
    }

    pub fn predict(&self, data: &Vec<f32>) -> Vec<f32> {
        data.iter().map(|x_value| exp_x(*x_value, self.c, self.k)).collect() 
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_exponential_regression() {
        let labels = vec![2.0, 4.0, 8.0, 16.0, 24.0];
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0] ;
        let mut exponential_regression = ExponentialRegression::new();
        exponential_regression.fit(&features, &labels, 300, 0.01);
        println!("{:?}", exponential_regression.predict(&features));
    } 
}
