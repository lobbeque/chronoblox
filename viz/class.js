/*
 * Class of elements to be drawn
 */ 

class canopyHull {
  constructor(period,points) {
    this.period = period;
    this.hull = getHull(points,40);
  }

  display(c) {  
      
      // draw the hole
      c.erase()
      c.push() 
      drawRoundedShape(this.hull,30,c)  
      c.pop()
      c.noErase()

      // draw the shadow
      c.push()
      c.strokeWeight(5)
      c.noFill();
      c.drawingContext.clip();
      c.drawingContext.shadowColor = 'black';
      c.drawingContext.shadowBlur = 20;
      drawRoundedShape(this.hull,30,c)
      c.drawingContext.shadowBlur = 0;
      c.pop()  
 
  }  

  getPeriod() {
    return this.period;
  }  
}


class taleEdge {
  constructor(x1,x2,y1,y2,tale_id) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.tale = tale_id;
  }  
  displayEdge(c,event) { 
    c.push();   
    if (event == "click") {
      c.stroke(taleColor[(taleClickCount - 1) % 2]);
    } else {
      c.stroke(taleColor[taleClickCount % 2]);
    }
    c.drawingContext.setLineDash([4, 6])
    c.strokeWeight(2.5)           
    c.line(this.x1,this.y1,this.x2,this.y2);    
    c.pop();
  }  
  getTale() {
    return this.tale;
  }  
}


class chronoSyncEdge {
  constructor(period,x1,x2,y1,y2,weight,source,target,id) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.w = syncEdgesScale(weight);
    this.period = period;
    this.source = source;
    this.target = target;
    this.id = id;
  }

  displayEdge(c,focused,event) {  
    let zoom = ((focused == "sync_edge_period_focus") ? 1 : 0)
    if (focused == "sync_edge_focus") {
      zoom += 1
    }    
    c.push();
    if (event == "click") {
      c.stroke(taleColor[(taleClickCount - 1) % 2]);
    } else if (event == "over") {
      c.stroke(taleColor[taleClickCount % 2]);
    } else {
      c.stroke(focusToColor(focused));
    }     
    c.strokeWeight(this.w + zoom);           
    c.line(this.x1,this.y1,this.x2,this.y2);
    c.pop();
  } 

  displayStroke(c,focused,event) { 
    c.push();
    c.stroke("#333333");
    let zoom = 1.5
    if (focused == "sync_edge_focus") {
      zoom += 2
    }    
    c.strokeWeight(this.w + zoom);           
    c.line(this.x1,this.y1,this.x2,this.y2);
    c.pop();
  } 

  getPeriod() {
    return this.period;
  }
  getWeight() {
    return this.w;
  } 
  getSource() {
    return this.source;
  }
  getTarget() {
    return this.target;
  }   
  getId() {
    return this.id;
  } 
}

function focusToColor(focused) {
  let focused_color;
  switch (focused) {
    case 'node_period_step':  
    case 'node_tale_step':  
    case 'node_neighbor_step':
    case 'node_over_step':
    case 'node_click_step':
      focused_color = color('#938A79');
      break;          
    case 'sync_edge_period_step':
      focused_color = color('#938A79');
      focused_color.setAlpha(100)
      break;
    case 'node_period_focus':  
    case 'node_tale_focus':  
    case 'node_neighbor_focus':
    case 'node_over_focus':
    case 'node_click_focus':
    case 'sync_edge_period_focus':
      focused_color = color("#FDEA24");
      break;                
    case 'sync_edge_focus':
      focused_color = color("#5cc9f5");
      break;                   
  } 
  return focused_color;   
} 

class chronoNode {
  constructor(id,x,y,weight,tale_id,meta,label) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.w = nodesScale(weight);
    this.period = id.split('_')[1];
    this.tale = tale_id;
    this.meta = meta;
    this.label = label;
  }

  isOver() {
    if ((dist(mouseX,mouseY,this.x,this.y + h/13) <= this.w / 2)) {
      return true;
    } else {
      return false;
    }
  }

  isClicked() {
    if (this.isOver()) {
      lastNodeClicked = this.id;
      lastNodeClickedNeighbors = []
      taleClickCount += 1;
      taleClicked = this.tale;
    } 
  }

  getMetaColor() {
    if (this.meta != "mixed") {
      return color(metaColorScale(this.meta))
    } else {
      return color(mixedMeta)
    }
  }

  drawLabel(c,span,label) {
    c.strokeWeight(3);
    c.stroke("#333333");
    c.textSize(20);
    c.textAlign(CENTER);
    c.textFont(neuekabelbold)
    c.fill(color("#FFFFFF"))
    c.text(label, this.x, this.y - span);
  }

  display(c,focused,event) {

    if (this.isOver()) {
      isAnyNodeOver = true;
      lastNodeOver = this.id;
      lastNodeOverNeighbors = [];
    }

    let fill_color = focusToColor(focused)
    let zoom = 0

    if (displayMeta) {
      fill_color = this.getMetaColor()
    }

    c.push()
    c.stroke("#333333");
    c.strokeWeight(1);
    if (focused == "node_period_focus") {
      c.strokeWeight(3);
    }

    if (focused == "node_neighbor_step" || focused == "node_neighbor_focus") {
      c.strokeWeight(3);
      c.stroke(taleColor[(taleClickCount - 1) % 2]);
      c.fill("#333333");
      c.circle(this.x ,this.y, this.w + zoom + 5);
      c.noStroke();      
    }

    if (focused == "node_click_focus" || focused == "node_click_step" || focused == "node_over_focus" || focused == "node_over_step") {
      c.strokeWeight(3);
      if (event == "click") {
        c.stroke(taleColor[(taleClickCount - 1) % 2]);
      } else {
        c.stroke(taleColor[taleClickCount % 2]);
      }
      c.fill("#333333");
      c.circle(this.x ,this.y, this.w + zoom + 5);
      c.noStroke();
    }  


    if (focused == "node_tale_step" || focused == "node_tale_focus") {
      c.strokeWeight(2.5);
      if (event == "click") {
        c.stroke(taleColor[(taleClickCount - 1) % 2]);
      } else {
        c.stroke(taleColor[taleClickCount % 2]);
      }
      c.fill("#333333");
      c.drawingContext.setLineDash([4, 6])
      c.circle(this.x ,this.y, this.w + zoom + 5);
      c.noStroke();
    }  

    if (periods.indexOf(this.period) == focus) {
      fill_color = color("#FDEA24")
    }  

    c.fill(fill_color);    
    c.circle(this.x ,this.y, this.w + zoom);
    if (this.isOver() || this.id == lastNodeClicked) {
      this.drawLabel(c,this.w + zoom,this.label)
    }
    c.pop();
    
  }

  getPeriod() {
    return this.period;
  }
  getWeight() {
    return this.w;
  }
  getId() {
    return this.id;
  } 
  getTale() {
    return this.tale;
  } 

}