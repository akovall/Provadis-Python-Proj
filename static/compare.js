// Original-vs-SVG comparison with a split handle, click-to-zoom and panning,
// shared by the converter and the gallery viewer. The markup is compareMarkup() below.

const COMPARE_ZOOM = 4;

function compareMarkup() {
  return `
    <div class="layer"><img class="original-image" alt="Original image"></div>
    <div class="layer vector"><img class="vector-image" alt="Vectorized image"></div>
    <div class="divider"><span>↔</span></div>
    <span class="tag original">Original</span>
    <span class="tag svg">SVG</span>
    <span class="tag hint"></span>`;
}

// `freeSpace()` returns the {width, height} the comparison may fill.
function createCompare(compare, freeSpace) {
  compare.innerHTML = compareMarkup();
  const original = compare.querySelector(".original-image"), vector = compare.querySelector(".vector-image");
  const vectorLayer = vector.parentElement, divider = compare.querySelector(".divider");
  const hint = compare.querySelector(".hint");
  let baseWidth = 0, baseHeight = 0, fit = 1, zoom = 1, offsetX = 0, offsetY = 0;
  let drag = null, splitting = false;

  function setSplit(value) {
    const split = Math.min(Math.max(value, 0), 1);
    vectorLayer.style.clipPath = `inset(0 0 0 ${split * 100}%)`;
    divider.style.left = `${split * 100}%`;
  }

  function layout() {
    if (!original.naturalWidth || !compare.offsetParent) return;
    const space = freeSpace();
    // Fill the free space even when the image is small, so its pixels are visible right away
    fit = Math.min(space.width / original.naturalWidth, space.height / original.naturalHeight);
    baseWidth = Math.round(original.naturalWidth * fit);
    baseHeight = Math.round(original.naturalHeight * fit);
    compare.style.width = baseWidth + "px";
    compare.style.height = baseHeight + "px";
    update();
  }

  function update() {
    const width = baseWidth * zoom, height = baseHeight * zoom;
    offsetX = Math.min(0, Math.max(baseWidth - width, offsetX));
    offsetY = Math.min(0, Math.max(baseHeight - height, offsetY));
    for (const img of [original, vector]) {
      Object.assign(img.style, { left: offsetX + "px", top: offsetY + "px", width: width + "px", height: height + "px" });
    }
    original.classList.toggle("pixelated", fit * zoom > 1);
    compare.classList.toggle("zoomed", zoom > 1);
    hint.textContent = zoom > 1 ? `×${COMPARE_ZOOM} · drag to move · click to zoom out` : "Click to zoom in";
  }

  function toggleZoom(event) {
    if (zoom > 1) {
      zoom = 1;
    } else {
      // Keep the clicked point under the cursor
      const rect = compare.getBoundingClientRect();
      offsetX = (event.clientX - rect.left) * (1 - COMPARE_ZOOM);
      offsetY = (event.clientY - rect.top) * (1 - COMPARE_ZOOM);
      zoom = COMPARE_ZOOM;
    }
    update();
  }

  function moveSplit(event) {
    const rect = compare.getBoundingClientRect();
    setSplit((event.clientX - rect.left) / rect.width);
  }

  function endDrag() {
    drag = null;
    compare.classList.remove("dragging");
  }

  original.addEventListener("load", () => { zoom = 1; layout(); });
  window.addEventListener("resize", layout);

  compare.addEventListener("pointerdown", (event) => {
    compare.setPointerCapture(event.pointerId);
    splitting = event.target.closest(".divider") !== null;
    drag = { x: event.clientX, y: event.clientY, offsetX, offsetY, moved: false };
  });
  compare.addEventListener("pointermove", (event) => {
    if (!drag) return;
    if (splitting) return moveSplit(event);
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    // A few pixels of movement still count as a click
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (drag.moved && zoom > 1) {
      compare.classList.add("dragging");
      offsetX = drag.offsetX + dx;
      offsetY = drag.offsetY + dy;
      update();
    }
  });
  compare.addEventListener("pointerup", (event) => {
    if (drag && !splitting && !drag.moved) toggleZoom(event);
    endDrag();
  });
  compare.addEventListener("pointercancel", endDrag);

  return { original, vector, setSplit, layout };
}
