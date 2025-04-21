use crate::dataframe::DataFrame;

pub trait Scalar {
    fn scale_data(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame;
} 