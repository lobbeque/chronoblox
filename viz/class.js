/*
 * Class of elements to be drawn
 */ 

class canopyHull {
  constructor(period,points) {
    this.period = period;
    if (points.length > 1) {
      this.hull = getHull(points,40);
    } else {
      this.hull = points;
    }
  }

  display(c) {  
    if (this.hull.length > 1) {
      // draw the hole
      c.erase()
      c.push() 
      drawRoundedShape(this.hull,30,c)  
      c.pop()
      c.noErase()
      // draw the shadow
      c.push()
      c.strokeWeight(5)
      c.stroke("#ff6361")
      c.noFill();
      c.drawingContext.clip();
      c.drawingContext.shadowColor = 'black';
      c.drawingContext.shadowBlur = 20;
      drawRoundedShape(this.hull,30,c)
      c.drawingContext.shadowBlur = 0;
      c.pop() 
    } else {
      // draw the hole
      c.erase()
      c.push() 
      c.circle(this.hull[0].x,this.hull[0].y,40)  
      c.pop()
      c.noErase()
      // draw the shadow
      c.push()
      c.strokeWeight(5)
      c.stroke("#ff6361")
      c.noFill();
      c.drawingContext.clip();
      c.drawingContext.shadowColor = 'black';
      c.drawingContext.shadowBlur = 20;
      c.circle(this.hull[0].x,this.hull[0].y,40) 
      c.drawingContext.shadowBlur = 0;
      c.pop()      
    } 
  }  
  getPeriod() {
    return this.period;
  }  
}

class flowEdge {
  constructor(period,x1,y1,x2,y2,alpha,lineage,lineage_color) {
    this.period = period;
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;        
    this.color = color("#261D11");
    this.color.setAlpha(alpha);
    this.lineage = lineage
    this.lineage_color = lineage_color
  }  
  display(c,focused) { 
    if (focused) {
      c.push();
      c.stroke("#ffffff");
      c.strokeWeight(6);           
      c.line(this.x1,this.y1,this.x2,this.y2);
      c.stroke(this.lineage_color);
      c.strokeWeight(4)           
      c.line(this.x1,this.y1,this.x2,this.y2);
      c.pop();
    } else {
      c.push()
      c.stroke(this.color)   
      c.strokeWeight(2);
      c.drawingContext.setLineDash([0.5, 4]);      
      c.line(this.x1,this.y1,this.x2,this.y2)
      c.pop()
    }
  }
  getPeriod() {
    return this.period;
  } 
  getWeight() {
    return this.w;
  } 
  getLineage() {
    return this.lineage;
  }             
}

class flowNode {
  constructor(id,x,y,weight,tale_id,tale_color) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.w = weight;
    this.period = id.split('_')[1];
    this.tale = tale_id;
    this.tale_color = tale_color;
  }
  display(c) {  
    let fill_color = (((this.tale == currentTale) && (currentTale >= 0)) ? this.tale_color : "#261D11")
    let stroke_weight = (((this.tale == currentTale) && (currentTale >= 0)) ? 2 : 1)
    let stroke_color = (((this.tale == currentTale) && (currentTale >= 0)) ? "#ffffff" : "#e5e1d8")
    let zoom = ((this.period == periods[focus]) ? 5 : 0)
    c.push()
    c.stroke(stroke_color);
    c.strokeWeight(stroke_weight)
    c.fill(fill_color);
    c.circle(this.x,this.y,this.w + zoom);
    c.pop()
  }
  getPeriod() {
    return this.period;
  }
  getWidth() {
    return this.w;
  }
  getId() {
    return this.id;
  }
  getTale() {
    return this.tale;
  }      
}

class taleEdge {
  constructor(x1,x2,y1,y2,weight,tale_id,tale_color) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.w = talesEdgesScale(weight);
    this.tale = tale_id;
    this.tale_color = tale_color
  }
  displayStroke(c) { 
    c.push();   
    c.stroke("#ffffff");
    c.strokeWeight(this.w + 5)           
    c.line(this.x1,this.y1,this.x2,this.y2);    
    c.pop();
  }   
  displayEdge(c) { 
    c.push();   
    c.stroke(this.tale_color);
    c.strokeWeight(this.w)           
    c.line(this.x1,this.y1,this.x2,this.y2);    
    c.pop();
  }  
  getTale() {
    return this.tale;
  }  
}


class chronoSyncEdge {
  constructor(period,x1,x2,y1,y2,weight) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.w = chronoSyncEdgesScale(weight);
    this.period = period;
  }

  displayEdge(c,focused) { 
    let fill_color = focusToColor(focused)
    let zoom = ((focused == "sync_edge_focus") ? 1 : 0)
    c.push();
    c.stroke(fill_color);
    c.strokeWeight(this.w + zoom);           
    c.line(this.x1,this.y1,this.x2,this.y2);
    c.pop();
  } 

  displayStroke(c) { 
    c.push();
    c.stroke("#ffffff");
    c.strokeWeight(this.w + 5);           
    c.line(this.x1,this.y1,this.x2,this.y2);
    c.pop();
  } 

  getPeriod() {
    return this.period;
  }
  getWeight() {
    return this.w;
  }  
}

function focusToColor(focused) {
  let focused_color;
  switch (focused) {
    case 'node_step':   
      focused_color = color(((shader == "night") ? '#A3A3A3' : '#938A79'));
      break;          
    case 'sync_edge_step':
      focused_color = color(((shader == "night") ? '#EFEFEF' : '#938A79'));
      focused_color.setAlpha(100)
      break;
    case 'node_focus': 
    case 'sync_edge_focus':
      focused_color = color("#0D0D0D");
      break;               
  } 
  return focused_color;   
} 

class nodeTale {
  constructor(x,y,weight,tale_id,tale_color) {
    this.x = x;
    this.y = y;
    this.w = nodesScale(weight);
    this.tale = tale_id;
    this.tale_color = tale_color;
  }
  display(c) {
    c.push()    
    c.stroke("#ffffff");
    c.strokeWeight(2);
    c.fill(this.tale_color);    
    c.circle(this.x ,this.y, this.w + 7);
    c.pop();
  }
  getTale() {
    return this.tale;
  } 
}

class chronoNode {
  constructor(id,x,y,weight,tale_id,meta) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.w = nodesScale(weight);
    this.period = id.split('_')[1];
    this.tale = tale_id;
    this.meta = meta;
  }

  isClicked() {
    if ((dist(mouseX,mouseY,this.x,this.y + h/5) <= this.w / 2) && (this.tale >= 0)) {
      displayTales = false;
      clickedNode = this.id;
      currentTale = this.tale;
      const btn_tales = document.getElementById("btntales")
      if (btn_tales.classList.contains("button-clicked")) {
        btn_tales.classList.remove('button-clicked');
      }        
    } 
  }

  display(c,focused) {

    let fill_color = focusToColor(focused)
    let zoom = ((focused == "node_focus") ? 1 : 0)

    if (displayMeta) {
      fill_color = metaColorScale(this.meta)
    }

    c.push()
    c.stroke("#ffffff");
    if (this.tale > -1) {
      c.drawingContext.setLineDash([0.5, 2]); 
      c.strokeWeight(1.5);
    } else {
      c.strokeWeight(1);
    }
    c.fill(fill_color);    
    c.circle(this.x ,this.y, this.w + zoom);
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