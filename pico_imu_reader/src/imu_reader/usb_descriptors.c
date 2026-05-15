// Minimal USB CDC descriptors for IMU reader (replaces pico_stdio_usb descriptors)

#include "tusb.h"
#include "pico/unique_id.h"

#define USBD_VID            0x2E8A  // Raspberry Pi
#define USBD_PID            0x000A  // CDC device
#define USBD_MANUFACTURER   "RPi"
#define USBD_PRODUCT        "IMU Reader"

#define USBD_ITF_CDC  0   // CDC needs 2 interfaces (comm + data)
#define USBD_ITF_MAX  2

#define USBD_CDC_EP_CMD  0x81
#define USBD_CDC_EP_OUT  0x02
#define USBD_CDC_EP_IN   0x82
#define USBD_CDC_CMD_MAX_SIZE    8
#define USBD_CDC_IN_OUT_MAX_SIZE 64

#define USBD_DESC_LEN (TUD_CONFIG_DESC_LEN + TUD_CDC_DESC_LEN)

// --- Device descriptor ---
static const tusb_desc_device_t desc_device = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = 0x0200,
    .bDeviceClass       = TUSB_CLASS_MISC,
    .bDeviceSubClass    = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol    = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor           = USBD_VID,
    .idProduct          = USBD_PID,
    .bcdDevice          = 0x0100,
    .iManufacturer      = 1,
    .iProduct           = 2,
    .iSerialNumber      = 3,
    .bNumConfigurations = 1,
};

// --- Configuration descriptor ---
static const uint8_t desc_configuration[] = {
    TUD_CONFIG_DESCRIPTOR(1, USBD_ITF_MAX, 0, USBD_DESC_LEN, 0, 250),
    TUD_CDC_DESCRIPTOR(USBD_ITF_CDC, 4, USBD_CDC_EP_CMD,
        USBD_CDC_CMD_MAX_SIZE, USBD_CDC_EP_OUT, USBD_CDC_EP_IN,
        USBD_CDC_IN_OUT_MAX_SIZE),
};

// --- String descriptors ---
static char serial_str[PICO_UNIQUE_BOARD_ID_SIZE_BYTES * 2 + 1];

static const char *const string_table[] = {
    [1] = USBD_MANUFACTURER,
    [2] = USBD_PRODUCT,
    [3] = serial_str,
    [4] = "IMU CDC",
};

// --- TinyUSB callbacks ---

const uint8_t *tud_descriptor_device_cb(void) {
    return (const uint8_t *)&desc_device;
}

const uint8_t *tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_configuration;
}

const uint16_t *tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;
    #define DESC_STR_MAX 20
    static uint16_t desc_str[DESC_STR_MAX];

    if (!serial_str[0]) {
        pico_get_unique_board_id_string(serial_str, sizeof(serial_str));
    }

    uint8_t len;
    if (index == 0) {
        desc_str[1] = 0x0409; // English
        len = 1;
    } else {
        if (index >= sizeof(string_table) / sizeof(string_table[0])) return NULL;
        const char *str = string_table[index];
        if (!str) return NULL;
        for (len = 0; len < DESC_STR_MAX - 1 && str[len]; ++len) {
            desc_str[1 + len] = str[len];
        }
    }

    desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * len + 2));
    return desc_str;
}
