// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DecentralizedEnergyMarket {
    address public regulator;
    uint256 public energyPool; // Total energy available for sale (in kWh)
    uint256 public energyPrice; // Price per unit of energy (in Wei)
    mapping(address => uint256) public energySold;
    mapping(address => uint256) public energyCredits; // Energy credits in Wei for each prosumer
    mapping(address => bool) public registeredProsumers;

    event EnergySold(address seller, uint256 amount, uint256 price);
    event EnergyBought(address buyer, uint256 amount, uint256 price);
    event ProsumerRegistered(address prosumer);
    event PriceUpdated(uint256 newPrice);
    event CreditsPurchased(address buyer, uint256 amount);

    constructor(uint256 _initialEnergy, uint256 _initialPrice) {
        regulator = msg.sender;
        energyPool = _initialEnergy; // Set initial energy pool
        energyPrice = _initialPrice; // Set initial price per unit
    }

    modifier onlyRegulator() {
        require(msg.sender == regulator, "Only the regulator can perform this action.");
        _;
    }

    function registerProsumer(address _prosumer) public onlyRegulator {
        registeredProsumers[_prosumer] = true;
        emit ProsumerRegistered(_prosumer);
    }

    function sellEnergy(uint256 _amount) public {
        require(registeredProsumers[msg.sender], "You must be a registered prosumer.");
        energyPool += _amount;
        energySold[msg.sender] += _amount;
        uint256 creditsEarned = _amount * energyPrice;
        energyCredits[msg.sender] += creditsEarned; // Earn credits in Wei for selling energy
        emit EnergySold(msg.sender, _amount, energyPrice);
    }

    function buyEnergy(uint256 _amount) public {
        require(registeredProsumers[msg.sender], "You must be a registered prosumer.");
        require(energyPool >= _amount, "Insufficient energy in the pool.");
        uint256 costInWei = _amount * energyPrice;
        require(energyCredits[msg.sender] >= costInWei, "Insufficient credits.");
        energyPool -= _amount;
        energyCredits[msg.sender] -= costInWei; // Spend credits in Wei to buy energy
        emit EnergyBought(msg.sender, _amount, energyPrice);
    }

    // Function to deposit Wei and receive energy credits
    function purchaseCredits() public payable {
        require(msg.value > 0, "You must send some Ether to purchase credits.");
        energyCredits[msg.sender] += msg.value; // Directly add the sent Wei to the user's credit balance
        emit CreditsPurchased(msg.sender, msg.value);
    }

    // Function for the regulator to update the energy price
    function updatePrice(uint256 _newPrice) public onlyRegulator {
        energyPrice = _newPrice;
        emit PriceUpdated(_newPrice);
    }
}
