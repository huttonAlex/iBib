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
| Compute | NVIDIA Jetson Orin Nano Super Developer Kit (8GB) | 1 | 249.00 | 249.00 | 67 TOPS, 2x CSI, JetPack/TensorRT (ORDERED) |
| Storage | Existing M.2 NVMe SSD 512GB (reused from RPi) | 1 | 0.00 | 0.00 | Must be NVMe (not SATA); 2230 or 2280 form factor |
| Camera | Arducam IMX477 MIPI CSI w/ CS mount (12.3MP) | 1 | 100.00 | 100.00 | Native JetPack driver, 4K30 / 1080p60, CS-mount |
| Lens | 8mm fixed focal length CS-mount lens | 1 | 40.00 | 40.00 | ~4m coverage at 5m; upgrade to varifocal if needed |
| GNSS | u-blox NEO-M8N breakout w/ PPS + SMA | 1 | 25.00 | 25.00 | PPS identical to M9N; UART or USB |
| Cellular | USB 4G LTE dongle (e.g. Huawei E3372) | 1 | 35.00 | 35.00 | Linux compatible; sufficient for CTP01 bandwidth |
| UPS Buffer | Mini UPS ~65W (e.g. PicoUPS-100 or Mylion) | 1 | 60.00 | 60.00 | Buffer for battery hot-swap only |
| Battery | Ryobi 18V ONE+ 6.0Ah + 9.0Ah 2-pack | 1 | 155.38 | 155.38 | Two batteries for hot-swap |
| Charger | Ryobi 18V charger | 1 | 34.97 | 34.97 | For battery recharge |
| Enclosure | Pelican 1200 case | 1 | 79.95 | 79.95 | IP-rated case |

**Subtotal (priced items): $779.30**

## TBD Items (Not Priced Yet)

- DC power distribution and wiring (fuse, ideal diode, connectors).
- Antennas and cabling (LTE + GNSS if not included in modem kit).
- Mounting hardware (tripod, quick-release plate, sun hood).
- Cooling parts (heatsink/fan included with dev kit, may need augmentation in enclosure).

## Sources

- NVIDIA Jetson Orin Nano Super Dev Kit: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- Arducam IMX477 for Jetson: https://www.arducam.com/embedded-camera-module/nvidia-jetson-nano-camera-arducam/nvidia-jetson-nano-nx-officially-supported-sensors/jetson-nano-nx-12mp-imx477-camera.html
- u-blox NEO-M8N breakout boards: https://gnss.store/en/neo-m8n-gnss-modules/44-elt0031.html
- Huawei E3372 USB LTE dongle: https://www.amazon.com/HUAWEI-E3372-Broadband-Unlocked-Network/dp/B013UURTL4
- PicoUPS-100 12V DC: https://www.mini-box.com/picoUPS-100-12V-DC-micro-UPS-system-battery-backup-system
- Ryobi 18V 9.0Ah battery: https://www.homedepot.com/p/RYOBI-18V-ONE-6-0-Ah-9-0-Ah-LITHIUM-HP-Battery-2-Pack-PBP2040/324026202
- Ryobi charger: https://www.homedepot.com/p/RYOBI-18V-ONE-Charger-P115/202981422
- Pelican 1200 case: https://www.pelican.com/us/en/product/cases/protector/1200/
