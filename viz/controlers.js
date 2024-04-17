/*
 * functions for user interaction 
 */


function setShader(mode) {
	shader = mode
	const btn_night = document.getElementById("nightshader");
	const btn_day = document.getElementById("dayshader");
	switch (mode) {
	  case 'night':
	  	btn_night.classList.add('button-clicked');
	  	btn_day.classList.remove('button-clicked');
	    break;
	  case 'day':
	  	btn_night.classList.remove('button-clicked');
	  	btn_day.classList.add('button-clicked');
	    break;
	}	
}

function removeTales() {
  const btn_tales = document.getElementById("btntales")
  if (btn_tales.classList.contains("button-clicked")) {
    btn_tales.classList.remove('button-clicked');
  } else {
    btn_tales.classList.add('button-clicked');
  }  
  if (!displayTales) {
    clickedNode = '';
    currentTale = -2; 
  }
  displayTales = ! displayTales 
}

function removeCanopy() {
  const btn_canopy = document.getElementById("btncanopy")
  if (btn_canopy.classList.contains("button-clicked")) {
    btn_canopy.classList.remove('button-clicked');
  } else {
    btn_canopy.classList.add('button-clicked');
  }
  displayCanopy = ! displayCanopy 
}

function removeMeta() {
  const btn_meta = document.getElementById("btnmeta")
  if (btn_meta.classList.contains("button-clicked")) {
    btn_meta.classList.remove('button-clicked');
  } else {
    btn_meta.classList.add('button-clicked');
  }
  displayMeta = ! displayMeta 
}

function removeEdges() {
  const btn_edges = document.getElementById("btnedges")
  if (btn_edges.classList.contains("button-clicked")) {
    btn_edges.classList.remove('button-clicked');
  } else {
    btn_edges.classList.add('button-clicked');
  }
  displayEdges = ! displayEdges 
}

function mouseClicked() {
  Object.values(chronoNodes).forEach((node) => {
    node.isClicked()
  })
}

function doubleClicked() {
  const btn_tales = document.getElementById("btntales")
  clickedNode = '';
  currentTale = -2;
  displayTales = false;
  if (btn_tales.classList.contains("button-clicked")) {
    btn_tales.classList.remove('button-clicked');
  }   
}

function keyPressed() {
  if (keyCode === RIGHT_ARROW) {
    if (focus != periods.length - 1) {
      focus += 1
    }
  } else if (keyCode === LEFT_ARROW) {
    if (focus != 0) {
      focus -= 1
    }
  }
} 