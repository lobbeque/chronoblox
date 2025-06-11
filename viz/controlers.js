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
      metaColorScale = metaColorScaleNight;
      break;
    case 'day':
      btn_night.classList.remove('button-clicked');
      btn_day.classList.add('button-clicked');
      metaColorScale = metaColorScaleLight;
      break;
  } 
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

function removeStepByStep() {
  const btn_step = document.getElementById("btnstepbystep")
  if (btn_step.classList.contains("button-clicked")) {
    btn_step.classList.remove('button-clicked');
  } else {
    btn_step.classList.add('button-clicked');
  }
  displayStepByStep = ! displayStepByStep 
}

function mouseClicked() {
  Object.values(chronoNodes).forEach((node) => {
    node.isClicked()
  })
}

function doubleClicked() {
  lastNodeClicked = '';  
  lastNodeClickedNeighbors = []  
}

function keyPressed() {
  if (keyCode === RIGHT_ARROW) {
    if (focus != periods.length - 1) {
      previous = focus
      focus += 1
    }
  } else if (keyCode === LEFT_ARROW) {
    if (focus != 0) {
      previous = focus
      focus -= 1
    }
  }
} 

function displayMouseOver(c) {
  if (lastNodeOver == lastNodeClicked) {
    return;
  }
  let node_over;
  // 0 : find the node under focus
  Object.values(chronoNodes).forEach((node) => {
    if (node.getId() == lastNodeOver) {
      node_over = node;
      return;
    }
  })

  // 1 : display the diachronic edges
  let taleId = node_over.getTale();
  let period = node_over.getPeriod()
  talesEdges[taleId].forEach((diac_edge) => {
    diac_edge.displayEdge(c,"over")
  })

  // 2 : find the synchronic nodes and display the synchronic edges
  sync_nodes_sources = []
  sync_nodes_targets = []
  if ((taleClicked != taleId) && (taleClicked != -1) && (taleId != -1)) {
    nodes_sources = talesNodes[taleId]
    nodes_targets = talesNodes[taleClicked]
    chronoSyncEdges.forEach((edge) => {
      source = edge.getSource()
      target = edge.getTarget()
      if ((nodes_sources.includes(source) || nodes_sources.includes(edge.getTarget()))
          && (nodes_targets.includes(source) || nodes_targets.includes(edge.getTarget()))) {
        edge.displayStroke(c,"sync_edge_focus","over")
        edge.displayEdge(c,"sync_edge_focus","over")
        if (nodes_sources.includes(source)) {
          sync_nodes_sources.push(source)
        }
        if (nodes_sources.includes(target)) {
          sync_nodes_sources.push(target)
        }
        if (nodes_targets.includes(source)) {
          sync_nodes_targets.push(source)
        }
        if (nodes_targets.includes(target)) {
          sync_nodes_targets.push(target)
        }        
      }    
    })
  }  

  // 3 : display the diachronic nodes
  talesNodes[taleId].forEach((id) => {
    diac_node = chronoNodes[id]
    diac_period = diac_node.getPeriod()
    if (id != node_over.getId()){
      if (periods.indexOf(diac_period) != focus) {
        diac_node.display(c,"node_tale_step","over")
      } else {
        diac_node.display(c,"node_tale_focus","over")
      }
    }
  })        

  // 5 : display the node under focus 
  sync_nodes_targets.forEach((node_id) => {
    Object.values(chronoNodes).forEach((node) => {
      if (node.getId() == node_id) {
        node.display(c,"node_neighbor_step","over")
      }
    })
  })

  sync_nodes_sources.forEach((node_id) => {
    Object.values(chronoNodes).forEach((node) => {
      if (node.getId() == node_id) {
        node.display(c,"node_over_step","over")
      }
    })
  })

  if (periods.indexOf(period) != focus) {
    node_over.display(c,"node_over_step","over")
  } else {
    node_over.display(c,"node_over_focus","over")
  }         
}

function displayMouseClick(c) {
  let node_click;
  // 0 : find the node under focus
  Object.values(chronoNodes).forEach((node) => {
    if (node.getId() == lastNodeClicked) {
      node_click = node;
      return;
    }
  })

  // 1 : display the diachronic edges
  let taleId = node_click.getTale();
  let period = node_click.getPeriod()
  talesEdges[taleId].forEach((diac_edge) => {
    diac_edge.displayEdge(c,"click")
  })

  // 2 : display the diachronic nodes
  talesNodes[taleId].forEach((id) => {
    diac_node = chronoNodes[id]
    diac_period = diac_node.getPeriod()
    if (id != node_click.getId()){
      if (periods.indexOf(diac_period) != focus) {
        diac_node.display(c,"node_tale_step","click")
      } else {
        diac_node.display(c,"node_tale_focus","click")
      }
    }
  }) 

  // 3 : display the node under focus      
  if (periods.indexOf(period) != focus) {
    node_click.display(c,"node_click_step","click")
  } else {
    node_click.display(c,"node_click_focus","click")
  }           
}