c
    }, {
      title: "Bottle in tray",
      source: "../../tasks/bottle_tray/rollouts/bottle_in_tray_demo.mp4"
    }, {
      title: "Flip open the lid",
      source: "../../tasks/lid_open/rollouts/lid_open_demo.mp4"
    }, {
      title: "Two-tier sort (long horizon)",
      source: "../../tasks/two_tier_sort/rollouts/two_tier_demo.mp4"onst videos = [
  {
    title: "Grasp demo",
    description: "The original 640 x 480 robot manipulation recording included with the project.",
    source: "../videos/grasp_demo_hd.mp4",
    meta: "640 x 480 · H.264 MP4"
  },
];

const grid = document.querySelector("#video-grid");
document.querySelector("#video-count").textContent = videos.length;

videos.forEach((video, index) => {
  const card = document.createElement("article");
  card.className = "video-card";
  card.innerHTML = `
    <div class="video-frame">
      <video controls preload="metadata" playsinline aria-label="${video.title}">
        <source src="${video.source}" type="video/mp4">
        Your browser does not support HTML5 video.
      </video>
    </div>
    <div class="video-copy">
      <div class="video-title-row">
        <span class="video-index">${String(index + 1).padStart(2, "0")}</span>
        <h3>${video.title}</h3>
      </div>
      <p>${video.description}</p>
      <span class="video-meta">${video.meta}</span>
    </div>
  `;
  grid.appendChild(card);
});
