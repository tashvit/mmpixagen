class CanvasTool {
    // This class is based on https://stackoverflow.com/a/52744318
    constructor(query) {
        this.query = query;
        this.canvasDom = $(query);
        this.canvas = this.canvasDom[0]
        this.ctx = this.canvas.getContext("2d")
        this.offsetX = 48;
        this.offsetY = 48;
        this.clearDraw();
        this.initEvents();
    }

    initState() {
        this.isMouseClicked = false;
        this.isMouseInCanvas = false;
        this.prevX = 0;
        this.currX = 0;
        this.prevY = 0;
        this.currY = 0;
        this.tool = "pen";
    }

    initEvents() {
        $(this.query).on("mousemove", (e) => this.onMouseMove(e));
        $(this.query).on("mousedown", (e) => this.onMouseDown(e));
        $(this.query).on("mouseup", () => this.onMouseUp());
        $(this.query).on("mouseout", () => this.onMouseOut());
        $(this.query).on("mouseenter", (e) => this.onMouseEnter(e));
    }

    onMouseDown(e) {
        this.isMouseClicked = true;
        this.updateCurrentPosition(e);
    }

    onMouseUp() {
        this.isMouseClicked = false;
    }

    onMouseEnter(e) {
        this.isMouseInCanvas = true;
        this.updateCurrentPosition(e);
    }

    onMouseOut() {
        this.isMouseInCanvas = false;
    }

    onMouseMove(e) {
        if (this.isMouseClicked && this.isMouseInCanvas) {
            this.updateCurrentPosition(e)
            this.draw()
        }
    }

    updateCurrentPosition(e) {
        this.prevX = this.currX;
        this.prevY = this.currY;
        this.currX = e.clientX - this.canvas.offsetLeft - this.offsetX;
        this.currY = e.clientY - this.canvas.offsetTop - this.offsetY;
    }

    draw() {
        this.ctx.beginPath()
        this.ctx.moveTo(this.prevX, this.prevY);
        this.ctx.lineTo(this.currX, this.currY);
        this.ctx.strokeStyle = (this.tool === "pen") ? "rgb(0,0,0)" : "rgb(255,0,255)";
        this.ctx.lineWidth = 4;
        this.ctx.stroke();
        this.ctx.closePath();
    }

    usePen() {
        this.tool = "pen";
    }

    useEraser() {
        this.tool = "eraser"
    }

    clearDraw() {
        this.initState();
        this.ctx.fillStyle = "rgb(255,0,255)"
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

var sketchCanvas;

function sketchUsePen() {
    sketchCanvas.usePen();
}

function sketchUseEraser() {
    sketchCanvas.useEraser();
}

function sketchClearCanvas() {
    sketchCanvas.clearDraw();
}

function genImage() {
    const b64Data = sketchCanvas.canvas.toDataURL();
    // Call API /gen
    $.ajax("/gen", {
        data: JSON.stringify({"image": b64Data, "task": "gen-image"}),
        contentType: "application/json",
        type: "POST",
        success: function (data) {
            const json = $.parseJSON(data);
            const img = new Image();
            img.onload = function () {
                document.getElementById("pixelArt").getContext("2d").drawImage(img, 0, 0);
            }
            img.src = json.image;
        }
    })
}

(function () {
    console.log("Creating sketch");
    sketchCanvas = new CanvasTool("#sketch");
    console.log(sketchCanvas.canvasDom);
    console.log(sketchCanvas.canvas);

    // On upload a sketch clicked
    document.getElementById("uploadSketch").onchange = function (e) {
        const img = new Image();
        img.onload = function draw() {
            sketchCanvas.ctx.drawImage(img, 0, 0, 256, 256);
        };
        const imageUrl = URL.createObjectURL(this.files[0]);
        console.log(imageUrl);
        img.src = imageUrl;
    };
})();
