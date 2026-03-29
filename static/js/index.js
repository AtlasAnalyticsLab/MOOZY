/* MOOZY Project Page — Interactivity */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCarousel();
  initEmbeddingToggles();
  initLightbox();
  initBibtexCopy();
});

/* --- Tabs --- */
function initTabs() {
  document.querySelectorAll(".tabs-component").forEach((component) => {
    const tabs = component.querySelectorAll(".tabs li");
    const panes = component.querySelectorAll(".tab-content");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.target;
        tabs.forEach((t) => t.classList.remove("is-active"));
        panes.forEach((p) => p.classList.remove("is-active"));
        tab.classList.add("is-active");
        document.getElementById(target).classList.add("is-active");
      });
    });
  });
}

/* --- Attention Map Carousel --- */
function initCarousel() {
  const container = document.querySelector(".carousel-container");
  if (!container) return;

  const track = container.querySelector(".carousel-track");
  const items = track.querySelectorAll("img");
  const dots = container.parentElement.querySelectorAll(".dot");
  const prevBtn = container.querySelector(".carousel-btn.prev");
  const nextBtn = container.querySelector(".carousel-btn.next");
  let current = 0;
  const total = items.length;

  function goTo(idx) {
    current = ((idx % total) + total) % total;
    track.style.transform = `translateX(-${current * 100}%)`;
    dots.forEach((d, i) => d.classList.toggle("active", i === current));
    // Update caption
    const captions = container.parentElement.querySelectorAll(".carousel-caption");
    captions.forEach((c, i) => (c.style.display = i === current ? "block" : "none"));
  }

  prevBtn.addEventListener("click", () => goTo(current - 1));
  nextBtn.addEventListener("click", () => goTo(current + 1));
  dots.forEach((dot, i) => dot.addEventListener("click", () => goTo(i)));

  // Keyboard
  container.setAttribute("tabindex", "0");
  container.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") goTo(current - 1);
    if (e.key === "ArrowRight") goTo(current + 1);
  });

  // Touch swipe
  let touchStartX = 0;
  container.addEventListener("touchstart", (e) => {
    touchStartX = e.touches[0].clientX;
  });
  container.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 50) {
      goTo(dx > 0 ? current - 1 : current + 1);
    }
  });

  goTo(0);
}

/* --- Embedding Visualization Toggles --- */
function initEmbeddingToggles() {
  const section = document.getElementById("embeddings-section");
  if (!section) return;

  const methodBtns = section.querySelectorAll(".toggle-method");
  const taskBtns = section.querySelectorAll(".toggle-task");
  const images = section.querySelectorAll(".emb-img");

  let method = "umap";
  let task = "cptac_cancer_type";

  function update() {
    images.forEach((img) => {
      const encoder = img.dataset.encoder;
      img.src = `static/images/embeddings/${method}_${task}_${encoder}.webp`;
    });
  }

  methodBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      methodBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      method = btn.dataset.method;
      update();
    });
  });

  taskBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      taskBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      task = btn.dataset.task;
      update();
    });
  });
}

/* --- Lightbox --- */
function initLightbox() {
  const overlay = document.getElementById("lightbox");
  if (!overlay) return;
  const lbImg = overlay.querySelector("img");

  document.querySelectorAll(".fig-clickable").forEach((img) => {
    img.addEventListener("click", () => {
      lbImg.src = img.dataset.full || img.src;
      overlay.classList.add("is-active");
    });
  });

  overlay.addEventListener("click", () => overlay.classList.remove("is-active"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") overlay.classList.remove("is-active");
  });
}

/* --- BibTeX Copy --- */
function initBibtexCopy() {
  const btn = document.querySelector(".copy-btn");
  if (!btn) return;
  const block = btn.parentElement.querySelector("code");

  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(block.textContent).then(() => {
      btn.textContent = "Copied!";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.classList.remove("copied");
      }, 2000);
    });
  });
}
