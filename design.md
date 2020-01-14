There are several “hash map” like data structures lie on a spectrum from high-performance, bare metal with restricted features to more convenient, 
full-featured structures that may be less performant. 
cuCollections will likely have several classes that are on different points on this spectrum.

# HashArray (TODO Naming)

Lowest-level, highest performance "bare metal" data structure with limited feature set.

## Keys and Values

- `sizeof(Key) + sizeof(Values) <= largest supported CAS (64bits)`

## Layout

### Array of Structs

- Enables 

### Struct of Arrays

## Operations



# HashMap (TODO Name)
