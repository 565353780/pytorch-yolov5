//	let detectURL = "https://api-oelink-sgmw-dev.servision.com.cn/detect";
// let detectURL = "https://127.0.0.1:8800/detect";
let detectURL = "127.0.0.1:8800/detect";
$(function()
{
  let audio = $("<audio  autoplay='autoplay' id='auto' ></audio>");
  $("body").append(audio)

  function playSound(src)
  {
    audio.attr("src",src);
  }

  let faceCnt = 0;
  var analysis = function(result)
  {
    let objects = result.reduce(function(s,v)
    {
      return s+v[2]
    },"");

    if (objects.indexOf("belt")<0)
    {
      faceCnt ++;
      if (faceCnt>2)
      {
        playSound('http://data.huiyi8.com/2017/gha/03/17/1702.mp3')
      }
    }
    else
    {
      faceCnt = 0;
    }

    $("#result").empty();
    const context = frame.getContext('2d');
    context.clearRect(0,0,$(canvas).outerWidth(),$(canvas).outerHeight());
    for (let i = 0; i < result.length; i++)
    {
      $("#result").append("<li>"+result[i][2]+"</li>");

      context.strokeStyle = {
        face:"red",
        wheel:"#777",
        belt:"black",
        hand:"#feffea",
        phone:"#bebf84",
      }[result[i][2]];
      context.beginPath();
      let r = result[i][0];
      if ($("#imgId").val()<5000 && $("#imgId").val()>4000)
      {
        let s = frame.width / frame.height;
        r = [r[1]*s,frame.height-r[0]/s,r[3]*s,frame.height-r[2]/s];
      }
      context.rect(r[0],r[1],r[2]-r[0],r[3]-r[1]);
      context.lineWidth = 3;
      context.stroke();
    }
  }

  function init()
  {
    if (navigator.mediaDevices === undefined)
    {
      navigator.mediaDevices = {};
    }
    if (navigator.mediaDevices.getUserMedia === undefined)
    {
      navigator.mediaDevices.getUserMedia = function (constraints)
      {
        var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia || navigator.oGetUserMedia;
        if (!getUserMedia)
        {
        }
        return new Promise(function (resolve, reject)
        {
          getUserMedia.call(navigator, constraints, resolve, reject);
        });
      }
    }
  }
  function showInfo()
  {
    if (navigator.mediaDevices)
    {
      $("#mediaDevices").html(JSON.stringify(navigator.mediaDevices));
    }
    else
    {
      $("#mediaDevices").css("color","red").html("undefined");
    }
    $("#appCodeName").html(navigator.appCodeName);
    $("#appMinorVersion").html(navigator.appMinorVersion);
    $("#appName").html(navigator.appName);
    $("#appVersion").html(navigator.appVersion);
    $("#cpuClass").html(navigator.cpuClass);
    $("#platform").html(navigator.platform);
  }

  function openCamera()
  {
    init();
    navigator.mediaDevices.getUserMedia(
    {
      audio: false,
      video: true
    }).then(function (mediaStream)
    {
      //将获取到的视频流放入video标签内展示
      if ("srcObject" in video)
      {
        video.srcObject = mediaStream;
      }
      else
      {
        video.src = window.URL.createObjectURL(mediaStream);
      }
      video.onloadedmetadata = function(e)
      {
        video.play();
        canvas.width = frame.width = video.offsetWidth;
        canvas.height = frame.height = video.offsetHeight;
      }

      start();
    }).catch(function (err)
    {
      console.log(err);
    })
  }

  var captureCnt = 0;
  var cnt = 0;

  var capture = function()
  {
    var xmlhttp = new XMLHttpRequest();
    if (xmlhttp.overrideMineType)
    {
      xmlhttp.overrideMineType("text/xml")
    }

    xmlhttp.open('POST', detectURL, true);
    xmlhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");

    let img = canvas.toDataURL("image/png").substring(22);
    xmlhttp.send(JSON.stringify({img}));
    xmlhttp.onreadystatechange = function ()
    {
      if (xmlhttp.readyState == 4 && xmlhttp.status == 200)
      {
        let result = JSON.parse(xmlhttp.responseText).Result;
        analysis(result);
      }
    };
  }

  var start = function()
  {
    setInterval(function()
    {
      canvas.getContext('2d').drawImage(video, 0 , 0 ,canvas.width, canvas.height);
    },200)

    var interval = setInterval(function(){capture()},2000/$("#time").val());
    $("#time").change(function()
    {
      clearInterval(interval);
      interval = setInterval(function(){capture()},2000/$("#time").val());
    })
  };

  openCamera();
  showInfo();

});

