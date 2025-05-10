

pub fn exp_x(x: f32, c: f32, k: f32) -> f32 {
    c * f32::exp(x * -k)  
}

pub fn residual(x: f32, y: f32, c: f32, k: f32) -> f32 {
    y - (c * f32::exp(-k * x))
}

pub fn residual_p_c(x: f32, k: f32) -> f32 {
    -f32::exp(-k * x) 
}

pub fn residual_p_k(x: f32, c: f32, k: f32) -> f32 {
    c * x * f32::exp(-k * x)
}