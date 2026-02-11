# Camera Placement Guide

How camera placement affects bib detection accuracy, and how to configure
PointCam for different venue setups.

---

## Quick Start

1. Choose a placement from the options below
2. Set `camera.placement` in your config (`"left"`, `"right"`, or `"center"`)
3. PointCam automatically adjusts bib crop padding to compensate for the viewing angle

---

## Placement Options

### Option A: Far Back, Elevated, Centered (Recommended)

```
         [CAMERA]   3-4m high, on tripod/stand
              \
               \    telephoto lens, ~20-30deg downward
                \
   ============ FINISH LINE ============

   runners approaching -->

   |<-- 15-25m back -->|
```

**Setup**: Tripod or light stand, 15-25m upstream of the finish line, centered on
the course, 3-4m high. Use a telephoto/zoom lens to fill the frame with the
finish zone.

**Config**: `camera.placement: "center"`

| Pros | Cons |
|------|------|
| Near head-on view of bibs (best OCR accuracy) | Needs telephoto lens for distance |
| Elevated = sees over front runners to the pack | Requires space behind finish area |
| Out of runners' way | Longer setup (tall stand + lens) |
| Works at most venues | |

**Best for**: Road races, large 5K/10K/marathon finishes, any venue where you have
clear sightline 15m+ behind the finish area.

---

### Option B: Side-Mounted, Elevated

```
                    [CAMERA]  2-3m high
                        |
                        |
   ============ FINISH LINE ============

   runners approaching -->
```

**Setup**: Tripod or clamped to a barrier/fence, 2-5m to the side of the finish
chute, elevated 2-3m. Can be mounted on timing equipment table, scaffolding,
or a tall tripod.

**Config**: `camera.placement: "left"` or `camera.placement: "right"`
(left/right is from the camera's perspective, facing the runners)

| Pros | Cons |
|------|------|
| Easy to set up alongside finish chute | Angled view causes digit truncation on far side |
| Uses existing infrastructure (tables, barriers) | Runners closer to camera occlude far-side runners |
| Compact setup | Lower accuracy than centered placement |

**Best for**: Small races, trail finishes, venues where you can't get behind the
finish line. Expect 10-20% lower bib accuracy than centered placement.

**Tip**: The further from the finish line (along the course axis), the better.
Being 3m to the side at 20m distance is much better than 3m to the side at 5m
distance (7 deg angle vs 31 deg).

---

### Option C: Truss / Overhead Mount

```
   ======== FINISH TRUSS ========
                |
            [CAMERA]   directly above or slightly upstream
                |
   ============ FINISH LINE ============
```

**Setup**: Camera mounted on the finish truss or overhead gantry, pointing
downward toward oncoming runners.

**Config**: `camera.placement: "center"`

| Pros | Cons |
|------|------|
| Centered, no left/right angle bias | Steep downward angle foreshortens bibs |
| Sees entire course width | Only works if truss is available |
| Completely out of the way | Bibs face forward, not up (bad if directly overhead) |

**Best for**: Venues with an existing finish truss, IF the camera can be mounted
slightly upstream (1-3m before the line) and angled toward oncoming runners
rather than pointing straight down.

**Warning**: A camera mounted directly above the finish line looking straight down
will see the tops of runners' heads, not their bibs. Offset upstream if possible.

---

## How Placement Affects the Pipeline

The `camera.placement` setting adjusts **bib crop padding** before OCR:

| Placement | Left Pad | Right Pad | Top Pad | Bottom Pad | Reason |
|-----------|----------|-----------|---------|------------|--------|
| `center` | 15% | 15% | 10% | 10% | Symmetric, slight extra horizontal for bib width |
| `left` | 10% | 25% | 10% | 10% | Far side (right) of bib is more likely clipped |
| `right` | 25% | 10% | 10% | 10% | Far side (left) of bib is more likely clipped |

When the camera is off to one side, the YOLO detector's bounding box tends to clip
the far edge of the bib (the side furthest from the camera). Extra padding on that
side gives the OCR model more context to read the outermost digit.

---

## Camera Selection Tips

- **Lens**: A tighter field of view (telephoto/zoom) is almost always better.
  Bibs should be at least 100px wide in the frame for reliable OCR. Wider is better.
- **Resolution**: 1920x1080 minimum. Higher resolution helps at distance.
- **Frame rate**: 30fps is sufficient. Higher fps helps with motion blur but
  increases processing load.
- **Mounting**: Secure mounting is critical. Wind, vibration from crowds, or
  accidental bumps will blur frames and lose detections.

---

## Lessons Learned (rec-0011 Test)

Our first field test used Option B (side-mounted, high-right, ~5m from finish):

- Detected 480 bibs in 30 minutes of a 5K finish
- 66% of detected bibs matched actual finishers
- Primary error: leading digit truncation (82% of phantom bibs were missing the
  first digit due to the camera angle clipping the left side of bibs)
- Secondary error: bibs physically crumpled/folded by running, making digits
  unreadable regardless of camera angle
- Bib crops were only ~85-135px wide in the 1920x1080 frame

**Key takeaway**: Distance + zoom beats close + wide-angle. Being further back
with a telephoto lens gives a more head-on view AND larger bib crops.
