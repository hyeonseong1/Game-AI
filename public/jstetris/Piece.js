class Piece{
    constructor(i){
        this.typeId = i;
        this.rotation = 0;
        this.create();
        this.lastMove = LAST_MOVE.NONE;
        this.rotTest = 0;
    }

    /**
     * Initializes the piece.
     */
    create = () =>{
        this.shape = PIECE_MAP[this.typeId][this.rotation];
        this.color = COLOR_MAP[this.typeId];
        this.initialPos();
        this.hardDropped = false;
    }

    /**
     * Marks the piece as hard-dropped.
     */
    hardDrop = () =>{
        this.hardDropped = true;
    }

    /**
     * Replaces this piece's state with the given piece.
     * Does nothing if the piece has been hard-dropped.
     * @param {Piece} p The piece to copy state from.
     */
    move = (p) => {
        if(this.hardDropped) return;
        this.x = p.x;
        this.y = p.y;
        this.shape = p.shape;
        this.color = p.color;
        this.rotation = p.rotation;
        this.lastMove = p.lastMove;
        this.rotTest = p.rotTest;
    }

    /**
     * Sets the piece to its initial spawn position.
     */
    initialPos = () =>{
        this.y  =   -1;
        this.x  =    3;
    }
}