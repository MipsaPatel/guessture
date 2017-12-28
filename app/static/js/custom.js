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
$("#upload-progress").slideUp(10)

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
    $("#video-form").delay(100).slideUp(250, function() {
        $("#upload-progress").slideDown(250, function() {
            $.ajax({
                url: '/guess',
                data: formData,
                method: 'POST',
                dataType: 'json',
                processData: false,
                contentType: false,
                success: function(data) {
                    $('#video-result').html(data['result'])
                    $("#upload-progress").delay(4000).slideUp(500, function() {
                        $('.result').fadeIn(1500)
                        $('html,body').stop().animate({scrollTop: $("#result").offset().top}, 1500, 'easeInOutBack')
                        $("#video-form").delay(1000).slideDown(2000)
                    })
                }
            })
        })
    })

}