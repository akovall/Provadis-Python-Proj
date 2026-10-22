// Saving an SVG as a PNG of a chosen size, shared by the converter and the gallery page.

const PNG_SCALES = [1, 2, 4, 8];
const MAX_PNG_SIDE = 8192;  // browsers refuse to draw larger canvases

function fillScaleSelect(select, width, height) {
  const previous = select.value || "2";
  select.innerHTML = "";
  for (const scale of PNG_SCALES) {
    const option = new Option(`×${scale} · ${width * scale}×${height * scale}`, scale);
    option.disabled = Math.max(width, height) * scale > MAX_PNG_SIDE;
    select.add(option);
  }
  const keep = [...select.options].find((option) => option.value === previous && !option.disabled);
  select.value = keep ? previous : select.options[0].value;
}

async function downloadPng(svgText, scale, fileName) {
  const svg = new DOMParser().parseFromString(svgText, "image/svg+xml").documentElement;
  const width = Math.round(parseFloat(svg.getAttribute("width")) * scale);
  const height = Math.round(parseFloat(svg.getAttribute("height")) * scale);
  // Draw the vector at the target size instead of stretching a small bitmap
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("the browser could not create the PNG");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = fileName;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } finally {
    URL.revokeObjectURL(url);
  }
}
