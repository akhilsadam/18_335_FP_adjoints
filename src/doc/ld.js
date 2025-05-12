const modeToggle = document.getElementById("toggle-input");
const labelToggle = document.getElementById("toggle-input-2");
var style = getComputedStyle(document.body)
const pl = style.getPropertyValue('--light-primary');
const pd = style.getPropertyValue('--dark-primary');
const sl = style.getPropertyValue('--light-secondary');
const sd = style.getPropertyValue('--dark-secondary');
const hd = style.getPropertyValue('--haiiro-d');
const hl = style.getPropertyValue('--haiiro-l');
const hll = style.getPropertyValue('--haiiro-ll');



const setElementStyle = (element, styles) => {
  Object.assign(element.style, styles);
};

// Event listeners

modeToggle.addEventListener("click", (event) => {
  if (modeToggle.checked) {
    setElementStyle(document.body, {
      backgroundColor: pd,
      color: pl,
    });
    document.documentElement.style.setProperty('--haiiro-ld', hd);
    document.documentElement.style.setProperty('--haiiro-ldu', hl);
    let el = Array.from(document.getElementsByTagName("video"));
    el.forEach((item) => {
      item.classList.add("dark-link");
      item.classList.remove("light-link");
    });
    let il = Array.from(document.querySelectorAll(".icon"));
    il.forEach((item) => {
      item.classList.add("dark-link");
      item.classList.remove("light-link");
    });
  } else {
    setElementStyle(document.body, {
      backgroundColor: sd,
      color: sl,
    });
    document.documentElement.style.setProperty('--haiiro-ld', hl);
    document.documentElement.style.setProperty('--haiiro-ldu', hll);
    let el = Array.from(document.getElementsByTagName("video"));
    el.forEach((item) => {
      item.classList.add("light-link");
      item.classList.remove("dark-link");
    });
    let il = Array.from(document.querySelectorAll(".icon"));
    il.forEach((item) => {
      item.classList.add("light-link");
      item.classList.remove("dark-link");
    });
  }
});

labelToggle.addEventListener("click", (event) => {
  if (labelToggle.checked) {
    document.documentElement.style.setProperty('--label-display', 'none');
  } else {
    document.documentElement.style.setProperty('--label-display', 'flex');
  }
});
