use std::io::{Cursor, Read};
use std::ops::Index;

pub struct CursorReader<'a> {
    cursor: Cursor<&'a [u8]>
}

impl<'a> CursorReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
        }
    }

    pub fn f32(&mut self) -> f32 {
        let mut b = [0u8; 4];
        self.cursor.read_exact(&mut b).unwrap();
        f32::from_be_bytes(b)
    }

    pub fn usize(&mut self) -> usize {
        let mut b = [0u8; 8];
        self.cursor.read_exact(&mut b).unwrap();
        usize::from_be_bytes(b)
    }

    pub fn i32(&mut self) -> i32 {
        let mut b = [0u8; 4];
        self.cursor.read_exact(&mut b).unwrap();
        i32::from_be_bytes(b)
    }

    pub fn indexed<T: From<usize>>(&mut self) -> T {
        T::from(self.usize())
    }

    pub fn cursor(self) -> Cursor<&'a [u8]> {
        self.cursor
    }

    pub fn pos(&self) -> usize {
        self.cursor.position() as usize
    }
}

pub struct VecWriter {
    vec: Vec<u8>
}

impl VecWriter {
    pub fn new() -> Self {
        Self {
            vec: Vec::new()
        }
    }

    pub fn vec(self) -> Vec<u8> {
        self.vec
    }

    pub fn f32(&mut self, v: f32) {
        self.vec.append(&mut v.to_be_bytes().to_vec());
    }

    pub fn usize(&mut self, v: usize) {
        self.vec.append(&mut v.to_be_bytes().to_vec());
    }

    pub fn i32(&mut self, v: i32) {
        self.vec.append(&mut v.to_be_bytes().to_vec());
    }

    pub fn index<T: Into<usize>>(&mut self, v: T) {
        let v= v.into();
        self.usize(v);
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }
}