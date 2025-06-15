# bitgrid

A fast, packed bit grid structure for managing N-Dim boolean matrices with:
- constant-time `get`/`set` 
- efficient iteration over set bits,
- arbitrary word sizes (`u8`, `u16`, `u32`, `u64`, `u128`), 
- `no_std` compatible

# Examples

```rust
use bitgrid::BitGrid2D;

// Create a 4x4 grid
let mut grid = BitGrid2D::<u32>::new(4, 4);

// Set some bits
grid.set(1, 2);
grid.set(3, 0);

// Check bit states
assert!(grid.get(1, 2));
assert!(!grid.get(0, 0));

// Iterate over set bits
let set_bits: Vec<_> = grid.iter().collect();
assert_eq!(set_bits, vec![[1, 2], [3, 0]]);
```

### For N-Dim grids:
``` rust
use bitgrid::BitGrid;

// create a 10x10x10x10 grid
let mut space = BitGrid::<u64, 4>::new_n([10, 10, 10, 10]);
space.set_n([1, 2, 3, 4]);
assert!(space.get_n([1, 2, 3, 4]));
```

