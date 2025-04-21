use crate::dataframe::DataFrame;

pub trait Encoder {
    fn apply_encoding(&self, df: &DataFrame, column_names: &Vec<String>) -> DataFrame;
}
