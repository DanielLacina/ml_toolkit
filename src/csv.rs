use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum DataTypeValue {
    Float(f32), 
    String(String)
} 

#[derive(Clone, Debug)]
pub enum DataType {
    Float,
    String
}

#[derive(Clone)]
pub struct DataFrame {
    index_to_column: HashMap<usize, String>,
    columns: HashMap<String, (usize, DataType)>, 
    data: Vec<Vec<DataTypeValue>>
} 

impl DataFrame {
    pub fn new() -> Self {
        return Self {
            index_to_column: HashMap::new(),
            columns: HashMap::new(),
            data: Vec::new()
        } 
    } 
    
    pub fn from_csv(filename: &str) -> Self {
        let mut df = DataFrame::new();
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file); 
        let mut reader_lines = reader.lines();
        let headers = reader_lines.next().unwrap().unwrap();
        df.insert_headers(&headers);
        for line in reader_lines {
            let line = line.unwrap(); 
            df.insert_row(&line); 
        } 
        return df;
    }

    fn insert_headers(&mut self, headers: &str) {
        for header in headers.split(",")  {
            self.insert_header(header);    
        }
    }

    fn insert_header(&mut self, header: &str) {
         let header_index = self.columns.len(); 
         self.columns.insert(header.to_string(), (header_index, DataType::Float));
         self.index_to_column.insert(header_index, header.to_string());
    } 

    fn insert_row(&mut self, line: &str) {
        let mut row = Vec::new();
        for (i, value) in line.split(",").enumerate() {
            let column = self.index_to_column.get(&i).unwrap();
            let (_, dtype) = self.columns.get(column).unwrap(); 
            match dtype {
                DataType::Float => {
                     match value.parse::<f32>() {
                        Ok(parsed_value) => {
                            row.push(DataTypeValue::Float(parsed_value));
                        } 
                        Err(_) => {
                            self.convert_columns_to_string(i);
                            row.push(DataTypeValue::String(value.to_string()));
                        } 
                     }
                },
                DataType::String => {
                     row.push(DataTypeValue::String(value.to_string())); 
                }
            }
        }
        self.data.push(row);
    }

    fn convert_columns_to_string(&mut self, i: usize) {
         for row in self.data.iter_mut() {
             let value = row[i].clone();
             match value {
                 DataTypeValue::Float(_float) => {
                     row[i] = DataTypeValue::String(_float.to_string());  
                 }
                 _ => {}
             }
         }
         let column_name = self.index_to_column.get(&i).unwrap();
         self.columns.insert(column_name.clone(), (i, DataType::String));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_from_csv() {
        let filename = "housing.csv";
        let df = DataFrame::from_csv(filename); 
        println!("{:?}", df.data);
    }
}