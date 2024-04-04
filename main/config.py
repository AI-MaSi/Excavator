
dir_path = "log/"
base_filename = "data_recording"

# identification numbers, you can add your own to the list
id_numbers = {
    0: "Excavator",
    1: "Mevea",
    2: "Motion Platform",
    3: "Digicenter"
}


# please don't change these yet
endian_specifier = '<' # '<'  Little-endian

unix_format = 'Q' # 'Q' 8 byte integer (UNIX-timestamp)

handshake_format = 'i' # 'i' Signed int (handshake) 4 bytes

# data_format = 'd' # 'd'  doubles (data) 8 bytes. Old Mevea way

data_format = 'b' # 'b' Signed int (data) 1 byte (For saving bandwidth)

checksum_format = 'B' # 'B'  Unsigned char (checksum) 1 byte