class random{
    constructor(){
        this.bag = 0x00;
        this.pieces = this.initPieces();
    }

    getPiece = (index) =>
    {
        while(this.pieces.length<index+8) this.addPiece();
        return this.pieces[index];
    }

    /**
     * Returns the next n pieces starting from the given index.
     * @param {int} n
     */
    nextPieces = (index) => this.pieces.slice(index,index+7);
    
    /**
     * Generates the initial bag of 7 pieces.
     */
    initPieces = () => {
        var tempArr =[]
        for(var i = 0; i<7;i++){
            tempArr.push(this.pullTet());
        }
        return tempArr;
    }

    /**
     * Adds the next piece to the queue.
     */
    addPiece = () => {
        this.pieces.push(this.pullTet());
    }

    /**
     * Draws a random piece from the current bag.
     * The bag resets after all 7 pieces have been drawn,
     * following the guideline 7-bag randomizer.
     */
    pullTet = () =>
    {
        do{
            var temp = parseInt(Math.random()*7);
        } while ((0x40>>temp ) & this.bag)

        this.bag = (this.bag | (0x40>>temp));

        this.checkBagFull();
        return temp;
    }

    /**
     * Resets the bag if all 7 pieces have been drawn.
     */
    checkBagFull = () =>
    {
        if(this.bag == 0x7f) // 0111 1111
            this.bag = 0x00
    }
}