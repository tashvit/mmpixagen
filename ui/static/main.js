class CanvasTool {
    // This class is based on https://stackoverflow.com/a/52744318
    constructor(query) {
        this.query = query;
        this.canvasDom = $(query);
        this.canvas = this.canvasDom[0]
        this.ctx = this.canvas.getContext("2d")
        this.offsetX = 0;
        this.offsetY = 0;
        this.lineWidth = 4;
        this.mirrorDraw = false;
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
        if (this.mirrorDraw) {
            const mirroredPrevX = this.canvas.width - this.prevX;
            const mirroredCurX = this.canvas.width - this.currX;
            // Left
            this.stroke(this.prevX, this.prevY, this.currX, this.currY);
            // Right
            this.stroke(mirroredPrevX, this.prevY, mirroredCurX, this.currY);
        } else {
            this.stroke(this.prevX, this.prevY, this.currX, this.currY);
        }

    }

    stroke(x1, y1, x2, y2) {
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = (this.tool === "pen") ? "rgb(0,0,0)" : "rgb(255,0,255)";
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.stroke();
        this.ctx.closePath();
    }

    usePen() {
        this.tool = "pen";
    }

    useEraser() {
        this.tool = "eraser"
    }

    toggleMirror() {
        this.mirrorDraw = !this.mirrorDraw;
    }

    clearDraw() {
        this.initState();
        this.ctx.fillStyle = "rgb(255,0,255)"
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

let sketchCanvas;

function sketchUsePen() {
    sketchCanvas.usePen();
}

function sketchUseEraser() {
    sketchCanvas.useEraser();
}

function sketchClearCanvas() {
    sketchCanvas.clearDraw();
}

function sketchSetBrushSize() {
    if (sketchCanvas.lineWidth === 4) {
        sketchCanvas.lineWidth = 16;
    } else {
        sketchCanvas.lineWidth = 4;
    }
}

function sketchCanvasSetMirror() {
    sketchCanvas.toggleMirror();
}

function setProgress(pr) {
    const progress = $("#progress");
    progress.width(pr + "%");
    progress.text(pr + "%");
}

function clearProgress() {
    const progress = $("#progress");
    const parent = $("#progressParent");
    parent.hide();
    parent.addClass("notransition");
    progress.width(0);
    progress.text("0%");
    parent.removeClass("notransition");
    parent.show();
}

function genImage() {
    clearProgress();
    const b64Data = sketchCanvas.canvas.toDataURL();
    // Call API /gen
    $.ajax("/gen", {
        data: JSON.stringify({"image": b64Data, "task": "gen-image"}),
        contentType: "application/json",
        type: "POST",
        success: function (data) {
            setProgress(50);
            const json = $.parseJSON(data);
            const img = new Image();
            img.onload = function () {
                document.getElementById("pixelArt").getContext("2d").drawImage(img, 0, 0);
                setProgress(100);
                setTimeout(clearProgress, 2000);
            }
            img.src = json.image;
        }
    })
}

function genSpriteSheet(model) {
    clearProgress();
    const canvasObject = document.getElementById("pixelArt");
    const b64Data = canvasObject.toDataURL();
    setProgress(20); // Show some progress until all work is done
    // Call API /gen
    $.ajax("/gen", {
        data: JSON.stringify({"image": b64Data, "task": "gen-sheet-" + model}),
        contentType: "application/json",
        type: "POST",
        success: function (data) {
            setProgress(80);
            const json = $.parseJSON(data);
            const img = new Image();
            img.onload = function () {
                document.getElementById("sheet").getContext("2d").drawImage(img, 0, 0);
                setProgress(100);
                setTimeout(clearProgress, 2000);
            }
            img.src = json.sheet;
        }
    })
}

function downloadCanvas(elementId) {
    const link = document.createElement('a');
    link.download = 'image-' + elementId + '.png';
    link.href = document.getElementById(elementId).toDataURL()
    link.click();
}


(function () {
    sketchCanvas = new CanvasTool("#sketch");

    // On upload a sketch clicked
    document.getElementById("uploadSketch").onchange = function (e) {
        const img = new Image();
        img.onload = function draw() {
            sketchCanvas.ctx.drawImage(img, 0, 0, 256, 256);
        };
        img.src = URL.createObjectURL(this.files[0]);
    };

    // On upload a pixel art clicked
    document.getElementById("uploadPixel").onchange = function (e) {
        const img = new Image();
        img.onload = function draw() {
            const canvas = $("#pixelArt")[0];
            canvas.getContext("2d").drawImage(img, 0, 0, 256, 256);
        };
        img.src = URL.createObjectURL(this.files[0]);
    };
})();
