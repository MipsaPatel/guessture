function collapseNavbar() {
    if($(".navbar").offset().top > 50) {
        $(".navbar-fixed-top").addClass("top-nav-collapse")
    }
    else {
        $(".navbar-fixed-top").removeClass("top-nav-collapse")
    }
}

$(window).scroll(collapseNavbar)
$(document).ready(collapseNavbar)
$(".result").fadeOut(10)
// TODO: add upload progress animation
//$("#upload-progress").fadeOut(10)

$(function() {
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this)
        $('html, body').stop().animate({scrollTop: $($anchor.attr('href')).offset().top}, 1000, 'easeInOutBack')
        event.preventDefault()
    })
    $('#video-file').on('change', fileUpload)
})

$('.navbar-collapse ul li a').click(function() {
    $(".navbar-collapse").collapse('hide')
})

function fileUpload(event) {
    $('#video-result').html('')
    $(".result").fadeOut(10)

    var file = event.target.files[0]
    var formData = new FormData($('#video-form')[0])
    $("#video-form").fadeOut(500)
    $.ajax({
        url: '/guess',
        data: formData,
        method: 'POST',
        dataType: 'json',
        processData: false,
        contentType: false,
        success: function(data) {
            $('#video-result').html(data['result'])
            $('.result').fadeIn(1500)
            $('html,body').stop().animate({scrollTop: $("#result").offset().top}, 1500, 'easeInOutBack')
            $("#video-form").fadeIn(5000)
        }
    })
}