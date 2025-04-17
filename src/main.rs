mod dataframe;
mod linear_regression;
mod matrices;
mod pipeline;
mod inference;
mod sampling;

use dataframe::DataFrame;
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
}
