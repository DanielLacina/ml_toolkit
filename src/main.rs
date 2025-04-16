mod dataframe;
mod linear_regression;
mod matrices;
mod pipeline;
use dataframe::DataFrame;
use linear_regression::LinearRegression;
use pipeline::{Pipeline, Scalar, StringEncoding, ImputerStrategy};

fn main() {
    let filename = "housing.csv";
    let df = DataFrame::from_csv(filename, None);
    let pipeline = Pipeline::new(StringEncoding::OneHot, ImputerStrategy::Median, Scalar::Standard); 
    let input_matrix = pipeline.transform(&df);
    let mut linear_regression = LinearRegression::new();
}
