Uses crc32 IEEE checksum on log headers. Segment files rolled at 64MB.
Crash recovery validates CRC on replay and truncates torn writes.