class Board{
    constructor(){
        this.field = this.initBoard();
        this.remaining = 0;
        this.garbage = 0;
    }

    /**
     * Initializes the board as a 2D array to store cell colors.
     * @return Empty 2D array
     */
    initBoard = () =>{
        var array = [];
        for(var i = 0;i<BOARD_HEIGHT;i++){
            array.push([]);
            for(var j = 0;j<BOARD_WIDTH;j++){
                array[i].push(0);
            }
        }
        return array;
    }

    isEmpty = () => this.remaining===0;
    /**
     * Locks the piece onto the board.
     * @param {Piece} p
     * @return {Number[]} Array of filled row indices
     */
    lock = p =>{
        for(var i = 0;i<4;i++){
            for(var j = 0; j<4;j++){
                if(p.shape & (0x8000 >> (i*4+j)))
                {
                    var tx = p.x + j;
                    var ty = p.y + i + 20;
                    this.field[ty][tx] = p.typeId+1;    
                }
            }
        }
        
        const data = {
            lines: [], 
            tSpin: T_SPIN_STATE.NONE,
            add: function(i) { this.lines.push(i); },
            get: function(i) { return this.lines[i]; },
            length: function() { return this.lines.length;}
        }
        
        var max = Math.min(p.y+24,BOARD_HEIGHT)
        for(var i = p.y+20; i<max; i++)
            if(this.checkLine(i)) data.add(i);

        this.remaining += 4;

        if(p.typeId===2 && p.lastMove === LAST_MOVE.SPIN)
            data.tSpin = this.checkTSpin(p.x, p.y, p.rotation,p.rotTest);
        
        return data;
    }

    /**
     * Checks whether the given row is completely filled.
     * @param {Number} y Row index
     * @return {boolean} True if filled
     */
    checkLine = y =>
    {
        let filled = true;
        for(var x = 0; x<BOARD_WIDTH;x++)
        {
            if(this.field[y][x]===0)
            {
                filled = false;
                continue;
            }
        }
        return filled;
    }

    /**
     * Clears the given row and shifts all rows above down by one.
     * @param {Number} i Row index
     */
    clearLine = i =>
    {
        for(var y = i;y>0;y--)
        {
            for(var x = 0; x<BOARD_WIDTH;x++)
            {
                this.field[y][x] = this.field[y-1][x];
            }
        }
        this.remaining -= 10;
    }

    /**
     * Checks whether the given cell is empty.
     * @param {Number} x Column index
     * @param {Number} y Row index
     * @return {boolean} True if the cell is empty
     */
    isNotBlocked = (x,y) =>
    {
        y = y+20;
        if(x<0||x>BOARD_WIDTH-1) return false;
        if(y>BOARD_HEIGHT-1) return false
        return this.field[y][x]===0;
    }

    /**
     * Checks whether the given piece can exist at its current position.
     * @param {Object} p - Piece to validate
     */
    valid = p =>
    {
        for(var i = 0;i<4;i++){
            for(var j = 0; j<4;j++){
                if(p.shape & (0x8000 >> (i*4+j)))
                {
                    var x = p.x + j;
                    var y = p.y + i;     
                    if(!this.isNotBlocked(x,y))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Checks whether the piece can move down one cell.
     * @param {Piece} p
     * @return {boolean} True if the piece can move down
     */
    canMoveDown = p =>
    {
        return this.valid({...p,y:p.y+1})
    }

    getRemaining = () =>
    {
        return this.remaining;
    }

    getGhostIndex = p =>
    {

        var temp = 0;
        while(this.canMoveDown(p))
        {
            p = {...p,y:p.y+1};
            temp++;
        }
        return temp;
    }
    
    checkTSpin = (x,y,r,l) =>
    {
        let corners = 0b0000;
        let tSpinCounter = 0;
        let tSpinMini = false;
        if(!this.isNotBlocked(x  ,y  )){        //LU
            corners = corners & 0b1000;
            tSpinCounter++;
        } 
        if(!this.isNotBlocked(x+2,y  ))       //RU
        {
            corners = corners & 0b0100;
            tSpinCounter++;
        }
        if(!this.isNotBlocked(x  ,y+2)) {     //LD
            corners = corners & 0b0010
            tSpinCounter++;
        }
        if(!this.isNotBlocked(x+2,y+2)){    //RD
            corners = corners & 0b0001
            tSpinCounter++;
        };

        if(tSpinCounter>2)
        {
            switch(r)
            {
                case 0:
                    tSpinMini = !(corners & 0b1100);
                    break;
                case 1:
                    tSpinMini = !(corners & 0b0101);
                    break;
                case 2:
                    tSpinMini = !(corners & 0b0011);
                    break;
                case 3:
                    tSpinMini = !(corners & 0b1010);
                    break;
            }
        }
        else return T_SPIN_STATE.NONE

        if(tSpinMini&&l<4) return T_SPIN_STATE.MINI;
        return T_SPIN_STATE.PROP
    }

    hardDrop = piece =>
    {
        let p = {...piece};
        let counter = 0;
        while(this.canMoveDown(p))
        {
            p.y++;
            counter++;
        }
        p.lastMove = LAST_MOVE.DOWN;
        return {
            piece: p,
            score: counter
        }
    }

    executeGarbage = () =>
    {
        let n = Math.min(this.garbage,BOARD_HEIGHT-1);
        for(let y = 0; y<BOARD_HEIGHT-n;y++)
            for(let x = 0; x<BOARD_WIDTH;x++)
                this.field[y][x] = this.field[y+n][x];

        let empty = parseInt(Math.random()*BOARD_WIDTH);
        for(let y = BOARD_HEIGHT-n;y<BOARD_HEIGHT;y++)
        {
            let chance = parseInt(Math.random()*BOARD_WIDTH);
            if(chance<3) empty = parseInt(Math.random()*BOARD_WIDTH);
            for(let x = 0; x<BOARD_WIDTH;x++)
                this.field[y][x] = (x==empty)?0:8;
        }
        this.garbage = 0;
    }

    addGarbage = n =>
    {
        this.garbage+=n;
    }
    
    deductGarbage = n =>
    {
        this.garbage -=n;
        if(this.garbage<0)
        {
            let a = 0 - this.garbage
            this.garbage = 0;
            return a;
        }
        return 0;
    }
}