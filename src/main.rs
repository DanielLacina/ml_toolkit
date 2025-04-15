mod linear_regression;
mod matrices;
mod csv;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

fn main() {
    let file = File::open("housing.csv").unwrap();
    let reader = BufReader::new(file); 
    let mut reader_lines = reader.lines();
    let headers = reader_lines.next().unwrap().unwrap();
    for header in headers.split(",") {
    } 
}
