Source: `NR/einsteintoolkit/repos/einsteinexamples/par/arXiv-1111.3344/bbh/BBHMedRes.par`

```ccl
IO::out_dir = $parfile

IOScalar::outScalar_every      = 32
IOScalar::one_file_per_group   = "yes"
IOScalar::outScalar_reductions = "minimum maximum average norm1 norm2"

IOASCII::one_file_per_group     = "yes"
IOASCII::out0D_every            = 32

IOHDF5::one_file_per_group            = "yes"
IOHDF5::open_one_input_file_at_a_time = "yes"
IOHDF5::out2D_every                   = 512
IOHDF5::out_every                     = 512000000

IOHDF5::checkpoint                  = "yes"
IO::checkpoint_dir                  = $parfile
IO::checkpoint_every                = 6144
IO::checkpoint_keep                 = 3
IO::checkpoint_on_terminate         = "yes"

IO::recover     = "autoprobe"
IO::recover_dir = $parfile
```

