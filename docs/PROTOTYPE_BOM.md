# Prototype BOM (Accuracy-First, Single Unit)

Date: 2026-02-05

Assumptions:
- 4K30 or 1080p60 capture with finish-line coverage (~4m chute at 5m distance).
- Minimum camera distance: 5 meters.
- Single unit, prototype accuracy-first (cost-down later).
- Hot-swappable power using Ryobi 18V batteries with a small UPS buffer.

Notes:
- Prices are list prices from vendor pages and will vary with availability and shipping.
- Camera module availability and Jetson Orin Nano Super driver support must be validated before purchase.
- Subtotal below excludes cables, mounts, and small hardware (TBD).

## BOM Table

| Item | Example Part | Qty | Unit Price (USD) | Extended (USD) | Notes |
| --- | --- | --- | --- | --- | --- |
| Compute | NVIDIA Jetson Orin Nano Super Developer Kit (8GB) | 1 | 249.00 | 249.00 | 67 TOPS, 2x CSI, JetPack/TensorRT |
| Storage | NVMe SSD 128GB (e.g. Kingston NV2) | 1 | 25.00 | 25.00 | Dev kit does not include storage |
| Camera | Arducam IMX477 MIPI CSI w/ CS mount (12.3MP) | 1 | 100.00 | 100.00 | Native JetPack driver, 4K30 / 1080p60, CS-mount |
| Lens | 8mm fixed focal length CS-mount lens | 1 | 40.00 | 40.00 | ~4m coverage at 5m; upgrade to varifocal if needed |
| GNSS | SparkFun GPS NEO-M9N SMA board | 1 | 76.50 | 76.50 | PPS for time sync |
| Cellular | Sixfab 4G/LTE Modem Kit (EC25-AF) | 1 | 129.00 | 129.00 | USB modem kit |
| UPS Buffer | 12V Mini UPS 65W | 1 | 90.00 | 90.00 | Buffer for hot-swap |
| Battery | Ryobi 18V ONE+ 6.0Ah + 9.0Ah 2-pack | 1 | 155.38 | 155.38 | Two batteries for hot-swap |
| Charger | Ryobi 18V charger | 1 | 34.97 | 34.97 | For battery recharge |
| Enclosure | Pelican 1200 case | 1 | 79.95 | 79.95 | IP-rated case |

**Subtotal (priced items): $980.80**

## TBD Items (Not Priced Yet)

- DC power distribution and wiring (fuse, ideal diode, connectors).
- Antennas and cabling (LTE + GNSS if not included in modem kit).
- Mounting hardware (tripod, quick-release plate, sun hood).
- Cooling parts (heatsink/fan included with dev kit, may need augmentation in enclosure).

## Sources

- NVIDIA Jetson Orin Nano Super Dev Kit: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- Arducam IMX477 for Jetson: https://www.arducam.com/embedded-camera-module/nvidia-jetson-nano-camera-arducam/nvidia-jetson-nano-nx-officially-supported-sensors/jetson-nano-nx-12mp-imx477-camera.html
- SparkFun NEO-M9N SMA board: https://www.sparkfun.com/products/15712
- Sixfab 4G/LTE Modem Kit (Jetson Nano): https://sixfab.com/product/4g-lte-modem-kit-for-jetson-nano/
- 12V Mini UPS 65W: https://www.nationalbatterysupply.com/12v-mini-ups-system-65w/
- Ryobi 18V 9.0Ah battery: https://www.homedepot.com/p/RYOBI-18V-ONE-6-0-Ah-9-0-Ah-LITHIUM-HP-Battery-2-Pack-PBP2040/324026202
- Ryobi charger: https://www.homedepot.com/p/RYOBI-18V-ONE-Charger-P115/202981422
- Pelican 1200 case: https://www.pelican.com/us/en/product/cases/protector/1200/
