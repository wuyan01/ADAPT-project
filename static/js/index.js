window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // Navbar burger toggle (kept in case a navbar is added later).
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  };

  bulmaCarousel.attach('.carousel', options);
  bulmaSlider.attach();

  // Attaching the carousel clones slides, so the lazy-video observer has to
  // pick up the copies it has not seen yet.
  if (window.adaptRefreshLazyVideos) {
    window.adaptRefreshLazyVideos();
  }
});
