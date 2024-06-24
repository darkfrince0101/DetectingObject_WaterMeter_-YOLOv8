var loader = document.querySelector(".loader");
const errorMessage = document.querySelector(".alert-danger");

window.addEventListener("load", () => {
  loader.classList.add("disappear");
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const loader = document.getElementById("loader");
  form.addEventListener("submit", () => {
    if (errorMessage) {
      errorMessage.style.display = "none";
    }
    // Show the loader
    loader.classList.remove("disappear");
  });
});
