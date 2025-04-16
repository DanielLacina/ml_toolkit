mod dataframe;
mod linear_regression;
mod matrices;
mod pipeline;
use dataframe::DataFrame;
use linear_regression::LinearRegression;
use pipeline::{Pipeline, Scalar, StringEncoding, ImputerStrategy};

fn main() {
    let filename = "housing.csv";
    let mut features_df = DataFrame::from_csv(filename, None);
    let labels_df = features_df.get_columns_as_df(&vec!["median_house_value".to_string()]); 
    features_df.remove_column("median_house_value");
    let features_pipeline = Pipeline::new(StringEncoding::OneHot, ImputerStrategy::Median, Scalar::Standard); 
    let inputs = features_pipeline.transform(&features_df);
    let labels_pipeline = Pipeline::new(StringEncoding::OneHot, ImputerStrategy::Median, Scalar::None);
    let labels = labels_pipeline.transform(&labels_df);
    let mut linear_regression = LinearRegression::new();
    linear_regression.fit(&inputs, &labels);
    println!("{:?}", linear_regression.weights());
    println!("{}", linear_regression.bias());
}
