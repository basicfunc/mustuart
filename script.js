const audioInput = document.getElementById("audioInput");

audioInput.addEventListener("change", function () {
  const file = audioInput.files[0];

  const reader = new FileReader();

  reader.onload = function (event) {
    const dataURL = event.target.result;
    localStorage.setItem("audioData", dataURL);
  };

  reader.readAsDataURL(file);

  console.log(file)
});
