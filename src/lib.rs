
#![cfg_attr(not(feature = "std"), no_std)]

//! A fast, packed bit grid structure for managing N-Dim boolean matrices with:
//! - constant-time `get`/`set` 
//! - efficient iteration over set bits,
//! - arbitrary word sizes (`u8`, `u16`, `u32`, `u64`, `u128`), 
//! - `no_std` compatible
//!
//! # Examples
//! ```
//! # use bitgrid::BitGrid2D;
//!
//! // Create a 4x4 grid
//! let mut grid = BitGrid2D::<u32>::new(4, 4);
//!
//! // Set some bits
//! grid.set(1, 2);
//! grid.set(3, 0);
//!
//! // Check bit states
//! assert!(grid.get(1, 2));
//! assert!(!grid.get(0, 0));
//!
//! // Iterate over set bits
//! let set_bits: Vec<_> = grid.iter().collect();
//! assert_eq!(set_bits, vec![[1, 2], [3, 0]]);
//! ```
//!
//! For N-Dim grids:
//! ```
//! # use bitgrid::BitGrid;
//!
//! // create a 10x10x10x10 grid
//! let mut space = BitGrid::<u64, 4>::new_n([10, 10, 10, 10]);
//! space.set_n([1, 2, 3, 4]);
//! assert!(space.get_n([1, 2, 3, 4]));
//! ```



extern crate alloc;

use alloc::{boxed::Box, vec::Vec};
use core::{fmt, ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr, Sub}};


/// Words are used as bit blocks in [`BitGrid`]
///
pub trait Word:
    Copy
    + Default
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + BitXor<Output = Self>
    + BitXorAssign
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + Not<Output = Self>
    + Sub<Output = Self>
    + PartialEq
    + Eq
{
    /// number of bits in word
    const N_BITS: u32;

    /// Words with all bits set to `1`
    const FULL: Self;

    /// Words with all bits set to `0`
    const EMPTY: Self;

    /// Words with least significant bit set to `1`
    const ONE: Self;

    /// Returns the number of trailing zeros
    fn trailing_zeros(self) -> u32;
}


macro_rules! impl_wrd {
    ($ty:ty) => {
        impl Word for $ty {
            const N_BITS: u32 = core::mem::size_of::<$ty>() as u32 * 8;
            const FULL: $ty = <$ty>::MAX;
            const EMPTY: $ty = 0;
            const ONE: $ty = 1;

            #[inline(always)]
            fn trailing_zeros(self) -> u32 {
                self.trailing_zeros()
            }
        }
    }
}

impl_wrd!(u8);
impl_wrd!(u16);
impl_wrd!(u32);
impl_wrd!(u64);
impl_wrd!(u128);

/// N-dimensional bit grid with packed storage
///
pub struct BitGrid<W, const N: usize> {
    /// Packed bit storage
    pub data: Box<[W]>,
    /// Grid dimensions [dim0, dim1, ..., dimN-1]
    pub shape: [u32; N],
    /// Total number of bits in the grid
    pub count: u32,
}

/// 2D bit grid specialization
pub type BitGrid2D<W> = BitGrid<W, 2>;
/// 3D bit grid specialization
pub type BitGrid3D<W> = BitGrid<W, 3>;

impl<WRD: Word, const N: usize> BitGrid<WRD, N> {

    /// Creates a new bit grid with the specified dimensions and all bits cleared
    ///
    /// # Panics
    /// Panics if the total grid size would overflow `u32`
    ///
    /// # Examples
    /// ```
    /// use bitgrid::BitGrid;
    /// let grid = BitGrid::<u32, 3>::new_n([10, 10, 10]);
    /// ```
    #[inline]
    pub fn new_n(shape: [u32; N]) -> Self {
        let mut count = 1;
        for d in shape {
            count *= d;
        }

        let n_words = ((count + WRD::N_BITS - 1) / WRD::N_BITS) as usize;
        let data = alloc::vec![WRD::EMPTY; n_words].into_boxed_slice();
        BitGrid { data, shape, count }
    }

    #[inline]
    pub fn flat_index(&self, coords: [u32; N]) -> (usize, u32) {
        let mut index = 0;
        let mut stride = 1;

        for (i, &dim) in self.shape.iter().enumerate().rev() {
            let coord = coords[i];
            debug_assert!(
                coords[i] < dim,
                "Coordinate {} out of bounds for dimension {} (size {})",
                coords[i],
                i,
                dim
            );
            index += coord * stride;
            stride *= dim;
        }

        let word = (index / WRD::N_BITS) as usize;
        let bit = index % WRD::N_BITS;
        (word, bit)
    }

    /// Sets the bit at the given coordinates
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds (debug builds only)
    ///
    /// # Examples
    /// ```
    /// use bitgrid::BitGrid;
    /// let mut grid = BitGrid::<u32, 2>::new_n([4, 4]);
    /// grid.set_n([1, 2]);
    /// ```
    #[inline]
    pub fn set_n(&mut self, coords: [u32; N]) {
        let (word, off) = self.flat_index(coords);
        self.data[word] |= WRD::ONE << off;
    }

    /// Clears the bit at the given coordinates
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds (debug builds only)
    #[inline]
    pub fn clear_n(&mut self, coords: [u32; N]) {
        let (word, off) = self.flat_index(coords);
        self.data[word] &= !(WRD::ONE << off);
    }

    /// Toggles the bit at the given coordinates
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds (debug builds only)
    pub fn flip_n(&mut self, coords: [u32; N]) {
        let (word, off) = self.flat_index(coords);
        self.data[word] ^= WRD::ONE << off;
    }

    /// Checks if the bit at given coordinates is set
    ///
    /// # Panics
    /// Panics if coordinates are out of bounds (debug builds only)
    ///
    /// # Examples
    /// ```
    /// use bitgrid::BitGrid;
    /// let mut grid = BitGrid::<u32, 2>::new_n([4, 4]);
    /// grid.set_n([1, 2]);
    /// assert!(grid.get_n([1, 2]));
    /// ```
    #[inline]
    pub fn get_n(&self, coords: [u32; N]) -> bool {
        let (word, off) = self.flat_index(coords);
        (self.data[word] & (WRD::ONE << off)) != WRD::EMPTY
    }

    /// Sets all bits in the grid to 1
    pub fn set_all(&mut self) {
        for w in &mut *self.data {
            *w = WRD::FULL;
        }
    }

    /// Sets all bits in the grid to 0
    pub fn clear_all(&mut self) {
        for w in &mut *self.data {
            *w = WRD::EMPTY;
        }
    }

    /// Returns an iterator over coordinates of all set bits
    ///
    /// The iterator yields coordinates in row-major order (last dimension changes fastest)
    ///
    /// # Examples
    /// ```
    /// # use bitgrid::BitGrid;
    /// 
    /// let mut grid = BitGrid::<u32, 2>::new_n([2, 2]);
    /// grid.set_n([0, 1]);
    /// grid.set_n([1, 0]);
    /// 
    /// let mut iter = grid.iter();
    /// assert_eq!(iter.next(), Some([0, 1]));
    /// assert_eq!(iter.next(), Some([1, 0]));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> BitIter<WRD, N> {
        if self.count == 0 {
            BitIter::default()
        } else {
            let last_i = (self.count - 1) / WRD::N_BITS;
            let valid  = self.count - last_i * WRD::N_BITS;
            let mask   = if valid == WRD::N_BITS { WRD::FULL } else {
                (WRD::ONE << valid) - WRD::ONE
            };
            let first  = self.data.get(0).cloned().unwrap_or(WRD::EMPTY)
                         & if last_i == 0 { mask } else { WRD::FULL };
            BitIter {
                data: &self.data,
                shape: self.shape,
                index: 0,
                curr_wrd: first,
            }
        }
    }

}

impl<WRD: Word> BitGrid<WRD, 2> {

    /// Creates a new 2D grid with specified dimensions
    pub fn new(x: u32, y: u32) -> Self {
        Self::new_n([x, y])
    }

    /// Gets the bit at (x, y)
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.get_n([x, y])
    }

    /// Sets the bit at (x, y)
    pub fn set(&mut self, x: u32, y: u32) {
        self.set_n([x, y]);
    }

    /// Clears the bit at (x, y)
    pub fn clear(&mut self, x: u32, y: u32) {
        self.clear_n([x, y]);
    }

    /// Toggle the bit at (x, y)
    pub fn flip(&mut self, x: u32, y: u32) {
        self.flip_n([x, y]);
    }

    /// Grid width (x-dimension)
    pub fn width(&self) -> u32 {
        self.shape[0]
    }

    /// Grid height (y-dimension)
    pub fn height(&self) -> u32 {
        self.shape[1]
    }
}

impl<W: Word> fmt::Display for BitGrid<W, 2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let ch = if self.get(x, y) { 'x' } else { 'o' };
                write!(f, "{}", ch)?;
            }
            if y + 1 < self.height() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}


impl<WRD: Word> BitGrid<WRD, 3> {

    /// Creates a new 3D grid with specified dimensions
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self::new_n([x, y, z])
    }

    /// Gets the bit value at (x, y, z)
    pub fn get(&self, x: u32, y: u32, z: u32) -> bool {
        self.get_n([x, y, z])
    }

    /// Sets the bit at (x, y, z)
    pub fn set(&mut self, x: u32, y: u32, z: u32) {
        self.set_n([x, y, z]);
    }

    /// Clears the bit at (x, y, z)
    pub fn clear(&mut self, x: u32, y: u32, z: u32) {
        self.clear_n([x, y, z]);
    }

    /// Toggles the bit at (x, y, z)
    pub fn flip(&mut self, x: u32, y: u32, z: u32) {
        self.flip_n([x, y, z]);
    }

    /// Grid width (x-dimension)
    pub fn width(&self) -> u32 {
        self.shape[0]
    }

    /// Grid height (y-dimension)
    pub fn height(&self) -> u32 {
        self.shape[1]
    }

    /// Grid depth (z-dimension)
    pub fn depth(&self) -> u32 {
        self.shape[2]
    }
}


/// Iterator over set bits in an N-dimensional [`BitGrid`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitIter<'a, W: Word, const N: usize> {
    /// Slice of words representing the bit field
    pub data: &'a [W],
    /// Dimensions of the grid [dim0, dim1, ..., dimN-1]
    pub shape: [u32; N],
    /// Current word index
    pub index: usize,
    /// Current word being scanned
    pub curr_wrd: W,
}

impl<'a, W: Word, const N: usize> Default for BitIter<'a, W, N> {
    fn default() -> Self {
        Self {
            data: &[],
            shape: [0; N],
            curr_wrd: W::EMPTY,
            index: 0,
        }
    }
}

impl<'a, W: Word, const N: usize> BitIter<'a, W, N> {
    /// Creates a new iterator for a grid with given shape and data
    pub fn new(shape: [u32; N], data: &'a [W]) -> Self {
        let first = data.first().copied().unwrap_or(W::EMPTY);
        Self {
            shape,
            data,
            index: 0,
            curr_wrd: first,
        }
    }

    /// Converts a flat bit index to N-dimensional coordinates
    ///
    fn crumple(&self, bit_index: u32) -> [u32; N] {
        let mut coords = [0; N];
        let mut remainder = bit_index;
        
        for i in (0..N).rev() {
            let stride = self.shape[i];
            coords[i] = remainder % stride;
            remainder /= stride;
        }
        
        coords
    }
}

impl<W: Word, const N: usize> Iterator for BitIter<'_, W, N> {
    type Item = [u32; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_wrd == W::EMPTY {
            self.index += 1;
            if self.index >= self.data.len() {
                return None;
            }
            self.curr_wrd = self.data[self.index];
        }

        let tz = self.curr_wrd.trailing_zeros();
        let bit_index = (self.index as u32) * W::N_BITS + tz;
        self.curr_wrd &= self.curr_wrd - W::ONE;

        Some(self.crumple(bit_index))
    }
}




#[cfg(test)]
mod test {
    use super::{BitGrid, BitGrid2D, BitGrid3D, Word};

    extern crate std;
    use std::{vec, vec::Vec};

    fn collect_bits<W: Word>(grid: &BitGrid2D<W>) -> Vec<[u32; 2]> {
        grid.iter().collect()
    }

    #[test]
    fn test_new_and_empty() {
        let g32 = BitGrid2D::<u32>::new(5, 4);
        assert_eq!(g32.width(), 5);
        assert_eq!(g32.height(), 4);
        assert!(g32.iter().next().is_none());

        let g64 = BitGrid2D::<u64>::new(7, 3);
        assert_eq!(g64.width(), 7);
        assert_eq!(g64.height(), 3);
        assert!(g64.iter().next().is_none());
    }

    #[test]
    fn test_set_and_get() {
        let mut g = BitGrid2D::<u32>::new(4, 4);
        g.set(0, 0);
        g.set(3, 0);
        g.set(1, 2);
        g.set(2, 3);

        assert!(g.get(0, 0));
        assert!(g.get(3, 0));
        assert!(g.get(1, 2));
        assert!(g.get(2, 3));
        assert!(!g.get(1, 1));
        assert!(!g.get(0, 3));
    }

    #[test]
    fn test_iterator_basic() {
        let mut g = BitGrid2D::<u64>::new(3, 3);
        for i in 0..3 {
            g.set(i, i);
        }
        let mut coords = collect_bits(&g);
        coords.sort_unstable();
        let expected = vec![[0,0], [1,1], [2,2]];
        assert_eq!(coords, expected);
    }

    #[test]
    fn test_iterator_sparse() {
        let mut g = BitGrid2D::<u32>::new(6, 2);
        g.set(5, 0);
        g.set(0, 1);
        g.set(3, 1);
        let mut coords = collect_bits(&g);
        coords.sort_unstable();
        let expected = vec![[0,1], [3,1], [5,0]];
        assert_eq!(coords, expected);
    }

    #[test]
    fn test_full_last_word_masking() {
        let w = 10;
        let h = 1;
        let mut g = BitGrid2D::<u32>::new(w, h);
        for x in 0..w {
            g.set(x, 0);
        }
        let coords = collect_bits(&g);
        assert_eq!(coords.len() as u32, w);
        let mut expected: Vec<_> = (0..w).map(|x| [x,0]).collect();
        expected.sort_unstable();
        assert_eq!(coords, expected);
    }

    #[test]
    fn packed_grid() {
        let mut g = BitGrid2D::<u32>::new(4, 4);
        g.set(1, 1);
        g.set(3, 1);
        g.set(3, 3);

        assert!(g.get(1, 1));
        assert!(g.get(3, 1));
        assert!(g.get(3, 3));

        let bits: Vec<_> = g.iter().collect();
        assert_eq!(&bits, &[[1, 1], [3, 1], [3, 3]], "{bits:?}");
    }


    #[test]
    fn test_set_clear_get() {
        let mut g = BitGrid2D::<u32>::new(4, 4);
        g.set(1, 2);
        assert!(g.get(1, 2));
        
        g.clear(1, 2);
        assert!(!g.get(1, 2));
        
        g.flip(1, 2);
        assert!(g.get(1, 2));
    }

    #[test]
    fn test_iterator() {
        let mut g = BitGrid2D::<u64>::new(3, 3);
        g.set(0, 0);
        g.set(1, 1);
        g.set(2, 2);
        
        let mut coords = collect_bits(&g);
        coords.sort();
        assert_eq!(coords, vec![[0, 0], [1, 1], [2, 2]]);
    }

    #[test]
    fn test_edge_cases() {
        // Empty grid
        let g = BitGrid2D::<u32>::new(0, 0);
        assert_eq!(g.iter().count(), 0);
        
        // Single cell grid
        let mut g = BitGrid2D::<u32>::new(1, 1);
        g.set(0, 0);
        assert_eq!(g.iter().collect::<Vec<_>>(), vec![[0, 0]]);
    }

    #[test]
    fn test_full_grid() {
        let mut g = BitGrid2D::<u32>::new(32, 2);
        for y in 0..2 {
            for x in 0..32 {
                g.set(x, y);
            }
        }
        assert_eq!(g.iter().count(), 64);
    }

    #[test]
    fn test_3d_grid() {
        let mut space = BitGrid3D::<u32>::new(2, 2, 2);
        space.set(0, 0, 0);
        space.set(1, 1, 1);
        
        let mut coords = space.iter().collect::<Vec<_>>();
        coords.sort();
        assert_eq!(coords, vec![[0, 0, 0], [1, 1, 1]]);
    }

    #[test]
    #[should_panic]
    fn test_overflow_protection() {
        let _ = BitGrid2D::<u32>::new(u32::MAX, u32::MAX);
    }

}

